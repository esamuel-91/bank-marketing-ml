from src.bank_marketing_ml.exception import BankMarketingException
from src.bank_marketing_ml import logger
from src.bank_marketing_ml.config.configuration import DataValidationConfig
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from src.bank_marketing_ml.utils.utils_main import save_json

import mlflow
import traceback

import pandas as pd
import sys


class DataValidation:
    def __init__(self, config: DataValidationConfig) -> None:
        self.config = config

    def gather_drift_report(self):
        """
        Fetches the reference and current dataset and use the evidently preset to generate the drift report.
        The generated report is then saved in json and html format to the drift path for future usage.

        Params:
            reference data: the data you are training the model
            current data: the data you want to test the model

        Raises:
            ValueError: if reference or current dataset is empty
                    or if columns mismatch or dtypes not match.
            BankMarketingException: if error occured while generating drift report

        Return:
            - drift status
        """
        try:
            logger.info("Loading reference and current dataset for drift analysis..")
            reference = pd.read_csv(self.config.train_path)
            current = pd.read_csv(self.config.test_path)

            if reference.empty or current.empty:
                raise ValueError("Reference or Current dataset must not be empty..")

            if set(reference.columns) != set(current.columns):
                raise ValueError(
                    f"Datasets columns does not match.. "
                    f"reference_columns: {sorted(reference.columns)}, "
                    f"current_columns: {sorted(current.columns)}"
                )

            num_columns = self.config.VALIDATION_SCHEMA.numerical_columns
            cat_columns = self.config.VALIDATION_SCHEMA.categorical_columns

            for column in set(num_columns) | set(cat_columns):
                if reference[column].dtype != current[column].dtype:
                    raise ValueError(
                        f"reference dtype: {reference.dtypes} is different from current dtype: {current.dtype}"
                    )

            report = Report(
                metrics=[
                    DataDriftPreset(drift_share=self.config.drift_share_threshold),
                    DataSummaryPreset(),
                ]
            )

            drift_report = report.run(reference_data=reference, current_data=current)
            logger.info("Saving the drift report into the artifacts folder..")
            self.config.drift_path.mkdir(parents=True, exist_ok=True)
            drift_report.save_html(str(self.config.drift_path / "drift_report.html"))
            result = drift_report.dict()

            drift_status = self._extract_drift_status(result)
            logger.info("Saving the drift status into the artifacts folder..")
            save_json(self.config.drift_path / "drift_status.json", drift_status)

            mlflow.log_artifact(
                str(self.config.drift_path / "drift_report.html"),
                artifact_path="drift_report",
            )

            mlflow.log_metric("drift_share", drift_status["drift_share"])
            mlflow.log_param("drift_detected", drift_status["drift_detected"])

            return drift_status

        except Exception as e:
            logger.exception("Exception occured while getting the drift status.")
            raise BankMarketingException(e, sys) from e

    def _extract_drift_status(self, result: dict) -> dict:
        for metric in result["metrics"]:
            share = metric["value"]["share"]
            count = metric["value"]["count"]

            return {
                "drift_detected": share >= self.config.drift_share_threshold,
                "drift_share": share,
                "drift_columns": count,
                "threshold": self.config.drift_share_threshold,
            }
        raise ValueError("DriftedColumnCount metric not found in the report.")

    def initiate_validation(self):
        """
        Validate training and testing datasets against the expected schema
        and save the validation report.
        """
        with mlflow.start_run(nested=True, run_name="Validating Data"):
            try:
                logger.info("Loading datasets for data validation..")
                train = pd.read_csv(self.config.train_path)
                test = pd.read_csv(self.config.test_path)

                dataframes = {"train": train, "test": test}

                schema = self.config.VALIDATION_SCHEMA
                expected_columns = schema.COLUMNS.keys()
                missing_rate = self.config.missing_rate

                errors = {}
                report = {}

                for dataframe_name, dataframe in dataframes.items():
                    dataframe_errors = []

                    if dataframe.empty:
                        dataframe_errors.append(f"{dataframe_name} must not be empty.")

                    if not dataframe.columns.is_unique:
                        duplicates = dataframe.columns[
                            dataframe.columns.duplicated()
                        ].to_list()
                        dataframe_errors.append(
                            f"{dataframe_name} has duplicate columns: {duplicates}"
                        )

                    actual_columns = set(dataframe.columns)
                    missing_columns = set(expected_columns) - actual_columns
                    extra_columns = actual_columns - set(expected_columns)

                    if missing_columns:
                        dataframe_errors.append(
                            f"{dataframe_name} has missing columns: {missing_columns}"
                        )

                    if extra_columns:
                        dataframe_errors.append(
                            f"{dataframe_name} has extra columns: {extra_columns}"
                        )

                    for column in expected_columns and actual_columns:
                        expected_dtype = schema.COLUMNS[column]
                        column_data = dataframe[column]

                        if not isinstance(column_data, pd.Series):
                            continue

                        actucal_dtype = column_data.dtype
                        if expected_dtype == "str":
                            dtype_matches = pd.api.types.is_string_dtype(
                                actucal_dtype
                            ) or pd.api.types.is_object_dtype(actucal_dtype)
                        else:
                            dtype_matches = actucal_dtype == pd.api.types.pandas_dtype(
                                expected_dtype
                            )

                        if not dtype_matches:
                            dataframe_errors.append(
                                f"Columns: {column} dtype mismatch "
                                f"expected dtype: {expected_dtype}, "
                                f"got: {actucal_dtype}"
                            )

                        missing_rate_dataframe = dataframe.isna().mean()
                        exceeded_rates = missing_rate_dataframe[
                            missing_rate_dataframe > missing_rate
                        ]

                        for column, rate in exceeded_rates.items():
                            dataframe_errors.append(
                                f"Columns: {column} missing rate exceeded "
                                f"expected missing rate: <= {missing_rate}, "
                                f"got: {rate:.2%}"
                            )

                    if dataframe_errors:
                        errors[dataframe] = dataframe_errors

                VALIDATION_STATUS = not errors

                for dataframe_name in dataframes:
                    report[dataframe_name] = {
                        "validation_status": dataframe_name not in errors,
                        "errors": errors.get(dataframe_name, []),
                    }

                logger.info("Saving the validation report into the artifacts folder.")
                self.config.STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)

                drift_status = self.gather_drift_report()

                validation_report = {
                    "validation_status": VALIDATION_STATUS,
                    "schema": report,
                    "drift": drift_status,
                }

                save_json(self.config.STATUS_FILE, validation_report)

                mlflow.log_artifact(
                    str(self.config.STATUS_FILE), artifact_path="validation_report"
                )

                if VALIDATION_STATUS:
                    mlflow.set_tag("Validation_Status", "Pass")
                else:
                    mlflow.set_tag("Validation_Status", "Fail")

                return {"schema": report, "drift": drift_status}

            except Exception as e:
                mlflow.set_tag("Validation_Status", "Fail")
                mlflow.log_text(traceback.format_exc(), "data_validation_traceback.txt")
                logger.exception("Data Validation error occured.")
                raise BankMarketingException(e, sys) from e
