from src.bank_marketing_ml.exception import BankMarketingException
from src.bank_marketing_ml import logger
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    MinMaxScaler,
    OrdinalEncoder,
    LabelEncoder,
)

from src.bank_marketing_ml.utils.common import save_arr, save_bin
from src.bank_marketing_ml.config.configuration import DataTransformationConfig

import mlflow
import traceback

import sys
import pandas as pd
import numpy as np


class DataTransformation:
    def __init__(self, config: DataTransformationConfig) -> None:
        self.config = config

    def gather_transformation_objs(self):
        """
        Builds and returns a preprocessing pipeline for numerical and categorical features.

        This method creates the necessary transformation pipelines for numerical
        and categorical features and combines them into a single preprocessor
        object for use in the machine learning workflow.

        Returns
        -------
        preprocessor : ColumnTransformer
            A preprocessor object that applies the appropriate transformations
            to numerical and categorical features.
        """
        try:
            logger.info(f"Fetching numerical and categorical columns.. ")

            numerical_columns = self.config.numerical_columns
            categorical_columns = self.config.categorical_columns

            if self.config.TRANSFORMATION_PARAMS.scaler == "standard":
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()

            num_pipline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy=self.config.TRANSFORMATION_PARAMS.num_imputer
                        ),
                    ),
                    ("scaler", scaler),
                ]
            )

            if self.config.TRANSFORMATION_PARAMS.encoder == "onehot":
                encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            else:
                encoder = OrdinalEncoder()

            cat_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy=self.config.TRANSFORMATION_PARAMS.cat_imputer
                        ),
                    ),
                    ("encoder", encoder),
                ]
            )

            logger.info("Creating preprocessor... ")
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor
        except Exception as e:
            logger.exception(
                f"Exception occured while creating the transformation pipeline."
            )
            raise BankMarketingException(e, sys) from e

    def initiate_transformation(self):
        """
        Initiates the data transformation process for training and testing datasets.

        Loads the training and testing data from the artifacts directory, separates
        the features and target variables, and applies the preprocessing pipeline
        and label encoding to transform the data.

        Saves the transformed datasets and associated artifacts to the artifacts
        directory and logs the transformation artifacts using MLflow.

        Returns
        -------
        None
            Saves the transformed data and logs the artifacts without returning
            a value.
        """
        with mlflow.start_run(nested=True, run_name="Transforming Data"):
            try:
                logger.info("Loading Train and Test dataset for transformation..")
                train_df = pd.read_csv(self.config.train_path)
                test_df = pd.read_csv(self.config.test_path)

                logger.info("Splitting feature and target columns..")
                input_feature_train_df = train_df.drop(self.config.target, axis=1)
                input_target_train_df = train_df[self.config.target]

                input_feature_test_df = test_df.drop(self.config.target, axis=1)
                input_target_test_df = test_df[self.config.target]

                logger.info(f"Transforming target using label encoder..")
                le = LabelEncoder()
                input_target_train_arr = le.fit_transform(input_target_train_df)
                input_target_test_arr = le.transform(input_target_test_df)

                logger.info(f"Saving le obj to the artifacts folder..")
                self.config.obj_path.mkdir(parents=True, exist_ok=True)
                save_bin(self.config.obj_path / "le.joblib", le)

                logger.info("Obtaining preprocessor object..")
                preprocessor_obj = self.gather_transformation_objs()

                logger.info(
                    "Transforming train and test features using transformating obj.."
                )
                input_feature_train_arr = preprocessor_obj.fit_transform(
                    input_feature_train_df
                )
                input_feature_test_arr = preprocessor_obj.transform(
                    input_feature_test_df
                )

                train_arr = np.c_[input_feature_train_arr, input_target_train_arr]
                test_arr = np.c_[input_feature_test_arr, input_target_test_arr]

                logger.info("Saving train and test arrays to the artifacts folder..")
                self.config.arr_path.mkdir(parents=True, exist_ok=True)
                save_arr(self.config.arr_path / "train_arr.npy", train_arr)

                save_arr(self.config.arr_path / "test_arr.npy", test_arr)

                logger.info(f"Saving preprocessor obj to the artifacts folder...")
                save_bin(self.config.obj_path / "preprocessor_obj", preprocessor_obj)

                mlflow.log_artifact(
                    str(self.config.arr_path / "train_arr.npy"),
                    artifact_path="transformed_arr",
                )
                mlflow.log_artifact(
                    str(self.config.arr_path / "test_arr.npy"),
                    artifact_path="transformed_arr",
                )

                mlflow.set_tag(
                    "Transformation_Status",
                    "Pass" if train_arr.size > 0 and test_arr.size > 0 else "False",
                )

            except Exception as e:
                mlflow.set_tag("Transformation_Status", "Fail")
                mlflow.log_text(
                    traceback.format_exc(), "data_transformation_traceback.txt"
                )
                logger.exception("Data Transformation Execution Failed.")
                raise BankMarketingException(e, sys)
