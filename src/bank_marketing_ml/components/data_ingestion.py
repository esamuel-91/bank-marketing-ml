from src.bank_marketing_ml.exception import BankMarketingException
from src.bank_marketing_ml import logger
from src.bank_marketing_ml.config.configuration import DataIngestionConfig
from src.bank_marketing_ml.config.mongodb_connection import MongoDBClient

import pandas as pd
import mlflow
import traceback
import sys

from sklearn.model_selection import train_test_split


class DataIngestion:
    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config

    def gather_collection_data(self):
        """
        Connects to the specific database and fetches the collection data as list and returns it.

        Raises:
            ValueError: if the collection is empty or if the collection name is none
            BankMarketingException: if error occured fetching the collection data as list
        Return:
            - data (list): collection data in list format
        """
        try:
            logger.info(
                f"Connecting the client to specific database: {self.config.database_name}"
            )
            client = MongoDBClient(database_name=self.config.database_name)

            if not self.config.collection_name:
                raise ValueError(f"Collection name must not be empty.")

            collection = client.database[self.config.collection_name]
            logger.info("Collecting data as list..")
            data = list(collection.find())

            if not data:
                raise ValueError(f"Collection: {self.config.collection_name} is empty!")

            return data

        except Exception as e:
            logger.exception("Exception occured while gathering the collection data.")
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
                logger.info("Fetching the collected data...")
                data = self.gather_collection_data()

                logger.info(f"Creating pd.DataFrame from the collected data..")
                df = pd.DataFrame(data)
                logger.info(
                    f"Shape of the dataframe; Rows: {df.shape[0]}, Columns: {df.shape[1]}"
                )
                logger.info(f"Saving the raw artifact into the artifacts folder..")
                self.config.feature_dir.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(self.config.feature_dir, index=False)

                logger.info("Dropping unwanted columns from the dataframe..")
                df.drop(self.config.drop_columns, axis=1, inplace=True)

                logger.info("Splitting the dataset into train and test dataset..")
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
                self.config.ingested_dir.mkdir(parents=True, exist_ok=True)
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

                if train.empty or test.empty:
                    raise ValueError("Train or Test dataset must not be empty...")

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
