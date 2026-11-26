from src.bank_marketing_ml.config.configuration import DataValidationConfig
from src.bank_marketing_ml.exception import BankMarketingException
from src.bank_marketing_ml import logger
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

from src.bank_marketing_ml.utils.common import save_json
import mlflow
import traceback
import sys

import pandas as pd


class DataValidation:
    def __init__(self, config: DataValidationConfig) -> None:
        self.config = config

    def gather_drift_status(self):
        """
        Loads the reference and current dataset for drift analysis.
        Saves the drift report and drift status into the artifacts folder.

        Raises:
            ValueError: if dataset is empty / column mismatch / dtype not same
            BankMarketingException: if error occured to get the drift status

        Return:
            - drift status
        """
        try:
            logger.info(
                f"Loading the reference and current dataset for drift analysis..."
            )
            reference_df = pd.read_csv(self.config.reference_path)
            current_df = pd.read_csv(self.config.current_path)

            dataframes = {
                "reference": reference_df,
                "current": current_df,
            }

            for dataframe_name, dataframe in dataframes.items():
                if dataframe.empty:
                    raise ValueError(f"{dataframe_name} must not be empty..")

            if set(reference_df.columns) != set(current_df.columns):
                raise ValueError(
                    f"Reference column: {sorted(reference_df.columns)} "
                    f"is different from current column: {sorted(current_df.columns)}"
                )

            numerical_columns = self.config.VALIDATION_SCHEMA.numerical_columns
            categorical_columns = self.config.VALIDATION_SCHEMA.categorical_columns

            for column in set(numerical_columns) | set(categorical_columns):
                if reference_df[column].dtype != current_df[column].dtype:
                    raise ValueError(
                        f"Column: {column} has dtype mismatch "
                        f"reference dtype: {reference_df[column].dtype} "
                        f"current dtype: {current_df[column].dtype}"
                    )

            report = Report(
                metrics=[
                    DataDriftPreset(drift_share=self.config.drift_share_threshold),
                    DataSummaryPreset(),
                ]
            )

            drift_report = report.run(
                reference_data=reference_df, current_data=current_df
            )
            logger.info("Saving the drift report into the artifacts folder..")
            self.config.drift_path.mkdir(parents=True, exist_ok=True)
            drift_report.save_html(str(self.config.drift_path / "drift_report.html"))

            result = drift_report.dict()
            drift_status = self._extract_drift_status(result)
            logger.info(f"Saving the drift status into the artifacts folder..")
            save_json(self.config.drift_path / "drift_status.json", drift_status)

            mlflow.log_artifact(
                str(self.config.drift_path / "drift_report.html"),
                artifact_path="drift_report",
            )
            mlflow.log_params(
                {"drift_share_threshold": self.config.drift_share_threshold}
            )

            return drift_status

        except Exception as e:
            logger.exception("Exception occured while getting the drift status.")
            raise BankMarketingException(e, sys) from e

    def _extract_drift_status(self, result: dict):
        for metric in result["metrics"]:
            share = metric["value"]["share"]
            count = metric["value"]["count"]
            return {
                "drift_detected": share >= self.config.drift_share_threshold,
                "drift_share": share,
                "drifted_columns": count,
                "drift_share_threshold": self.config.drift_share_threshold,
            }
        raise ValueError("DriftedColumnCount Metric not found in the report.")

    def initiate_validation(self):
        """
        Loads the train and test dataframe and validates the dataframe using VALIDATION SCHEMA.

        Return:
            - schema,report (dict)
        """
        with mlflow.start_run(nested=True, run_name="Validating Data"):
            try:
                logger.info(f"Loading the train and test dataset for validation...")
                train = pd.read_csv(self.config.reference_path)
                test = pd.read_csv(self.config.current_path)

                dataframes = {"train": train, "test": test}
                schema = self.config.VALIDATION_SCHEMA
                expected_columns = set(schema.COLUMNS.keys())
                missing_rate = self.config.missing_rate

                errors = {}
                report = {}

                for dataframe_name, dataframe in dataframes.items():
                    dataframe_errors = []
                    if dataframe.empty:
                        dataframe_errors.append(f"{dataframe_name} must not be empty.")

                    if not dataframe.columns.is_unique:
                        duplicate_columns = dataframe.columns[
                            dataframe.columns.duplicated()
                        ].to_list()

                        dataframe_errors.append(
                            f"{dataframe_name} has duplicate columns: {duplicate_columns}"
                        )

                    acutal_columns = set(dataframe.columns)
                    missing_columns = expected_columns - acutal_columns
                    extra_columns = acutal_columns - expected_columns

                    if missing_columns:
                        dataframe_errors.append(
                            f"{dataframe_name} has missing columns: {sorted(missing_columns)}"
                        )

                    if extra_columns:
                        dataframe_errors.append(
                            f"{dataframe_name} has extra columns: {sorted(extra_columns)}"
                        )

                    for column in expected_columns & acutal_columns:
                        expected_dtype = self.config.VALIDATION_SCHEMA.COLUMNS[column]
                        dataframe_column = dataframe[column]

                        actual_dtype = dataframe_column.dtype
                        if expected_dtype == "str":
                            dtype = pd.api.types.is_string_dtype(
                                actual_dtype
                            ) or pd.api.types.is_object_dtype(actual_dtype)
                        else:
                            dtype = actual_dtype == pd.api.types.pandas_dtype(
                                expected_dtype
                            )

                        if not dtype:
                            dataframe_errors.append(
                                f"{column} dtype mismatch: "
                                f"expected dtype: {expected_dtype} "
                                f"got: {actual_dtype}"
                            )

                    dataframe_missing_rate = dataframe.isna().mean()
                    exceeded_rate = dataframe_missing_rate[
                        dataframe_missing_rate > missing_rate
                    ]

                    for rate, column in exceeded_rate.items():
                        dataframe_errors.append(
                            f"Column: {column} has exceeded missing rate; "
                            f"expected: <= {missing_rate} "
                            f"got: {rate:.2%}"
                        )

                    if dataframe_errors:
                        errors[dataframe_name] = dataframe_errors

                VALIDATION_STATUS = not errors

                for dataframe in dataframes:
                    report[dataframe] = {
                        "Validation Status": dataframe not in errors,
                        "errors": errors.get(dataframe, []),
                    }

                logger.info(f"Saving the validation report to the artifacts folder..")
                self.config.STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)

                drift_status = self.gather_drift_status()
                validation_report = {
                    "Validation_Status": VALIDATION_STATUS,
                    "Schema": report,
                    "Drift_Status": drift_status,
                }

                save_json(self.config.STATUS_FILE, validation_report)

                mlflow.log_artifact(
                    str(self.config.STATUS_FILE), artifact_path="Validation Report"
                )

                mlflow.set_tag(
                    "Validation_Status", "Pass" if VALIDATION_STATUS else "False"
                )

                return {"Schema": report, "Drift": drift_status}

            except Exception as e:
                mlflow.log_text(traceback.format_exc(), "data_validation_traceback.txt")
                logger.exception(
                    "Exception occured while validating the train and test datasets."
                )
                raise BankMarketingException(e, sys) from e
