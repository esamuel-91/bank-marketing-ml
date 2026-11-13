from src.bank_marketing_ml.exception import BankMarketingException
from src.bank_marketing_ml import logger

from src.bank_marketing_ml.config.configuration import DataIngestionConfig
from src.bank_marketing_ml.config.mongodb_connection import MongoDBClient
from sklearn.model_selection import train_test_split
import pandas as pd

import mlflow
import traceback

import pandas as pd
import sys


class DataIngestion:
    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config

    def gather_collection_data(self):
        """
        Gathers the data from the specific collection under the database thats already connected

        Return:
            data (list): returns the data collected in list format
        Raises:
            ValueError: if the collection if empty
            BankMarketingException: if error occured while collecting the data from the collection
        """
        try:
            logger.info(
                f"Establishing connection to the specific database: {self.config.database_name}"
            )
            client = MongoDBClient(database_name=self.config.database_name)

            if not self.config.collection_name:
                raise ValueError("Collection name must not be empty..")

            collection = client.database[self.config.collection_name]
            data = list(collection.find())

            if not data:
                raise ValueError(f"Collection: {self.config.collection_name} is empty.")

            return data
        except Exception as e:
            logger.exception(
                f"Exception occured while collecting the data from specific collection: {self.config.collection_name}"
            )
            raise BankMarketingException(e, sys) from e

    def initiate_ingestion(self):
        """
        Fetches data from the database collection and converts it into a Pandas DataFrame.
        Splits the dataset into training and testing sets and saves the resulting artifacts
        in the artifacts directory.

        The ingestion pipeline uses a logger to track execution progress and record relevant
        metrics. MLflow is used to log the generated artifacts and evaluation metrics.

        Raises:
            BankMarketingException: If an error occurs during data ingestion or processing.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: The training and testing datasets.
        """
        with mlflow.start_run(nested=True, run_name="Ingesting Data"):
            try:
                logger.info("Getting the collected collection data as list.")
                data = self.gather_collection_data()

                logger.info("Creating dataframe from the collected data..")
                df = pd.DataFrame(data)
                logger.info(
                    f"Shape of the dataframe; Rows: {df.shape[0]}, Columns: {df.shape[1]}"
                )

                logger.info("Saving the raw artifact into the artifacts folder..")
                self.config.feature_dir.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(self.config.feature_dir, index=False)

                logger.info("Dropping the unwanted columns from the raw dataframe..")
                df.drop(columns=self.config.drop_columns, inplace=True)

                logger.info("Splitting the dataset into train and test")
                train, test = train_test_split(
                    df,
                    test_size=self.config.INGESTION_PARAMS.test_size,
                    random_state=self.config.INGESTION_PARAMS.random_state,
                    stratify=df[self.config.target]
                    if self.config.INGESTION_PARAMS.stratify
                    else None,
                )
                logger.info(
                    f"Shape of the train dataframe; Rows: {train.shape[0]}, Columns: {train.shape[1]}"
                )
                logger.info(
                    f"Shape of the test dataframe; Rows: {test.shape[0]}, Columns: {test.shape[1]}"
                )

                logger.info(
                    "Saving the train and test artifacts into the artifacts folder.."
                )
                self.config.ingested_dir.mkdir(exist_ok=True, parents=True)
                train.to_csv(
                    self.config.ingested_dir / "train.csv", index=False, header=True
                )
                test.to_csv(
                    self.config.ingested_dir / "test.csv", index=False, header=True
                )

                mlflow.log_artifact(
                    str(self.config.feature_dir), artifact_path="feature_store"
                )

                mlflow.log_artifact(
                    str(self.config.ingested_dir / "train.csv"),
                    artifact_path="ingested_data",
                )

                mlflow.log_artifact(
                    str(self.config.ingested_dir / "test.csv"),
                    artifact_path="ingested_data",
                )

                mlflow.log_metrics(
                    {
                        "test_size": self.config.INGESTION_PARAMS.test_size,
                        "random_state": self.config.INGESTION_PARAMS.random_state,
                    }
                )

                mlflow.log_params(
                    {
                        "database_name": self.config.database_name,
                        "collection_name": self.config.collection_name,
                        "stratify": self.config.INGESTION_PARAMS.stratify,
                    }
                )

                mlflow.set_tag(
                    "Ingestion_Status",
                    "Pass" if not train.empty or not test.empty else "Fail",
                )

                return train, test
            except Exception as e:
                mlflow.set_tag("Ingestion_Status", "Fail")
                mlflow.log_text(traceback.format_exc(), "data_ingestion_traceback.txt")
                logger.exception(
                    "Exception occured while ingesting the data from specific database."
                )
                raise BankMarketingException(e, sys) from e
