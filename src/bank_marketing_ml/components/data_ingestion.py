from src.bank_marketing_ml.exception import BankMarketingException
from src.bank_marketing_ml import logger
from src.bank_marketing_ml.config.configuration import DataIngestionConfig
from src.bank_marketing_ml.config.mongodb_connection import MongoDBClient

import mlflow
import traceback
import sys

import pandas as pd
from sklearn.model_selection import train_test_split


class Dataingestion:
    def __init__(self, config: DataIngestionConfig):
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
            client = MongoDBClient(database_name=self.config.database_name)
            logger.info("Checking if the collection name is set.")

            if not self.config.collection_name:
                raise ValueError(
                    f"Collection name must not be empty: {self.config.collection_name}"
                )
            collection = client.database[self.config.collection_name]
            data = list(collection)

            if not data:
                raise ValueError("Collection is empty..")
            else:
                return data

        except Exception as e:
            logger.exception(
                "Exception occured while collecting the documents in secific collection as list."
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
                logger.info("Loading collected data from the MongoDB collection.")
                data = self.gather_collection_data()
                df = pd.DataFrame(data)
                logger.info(
                    f"Shape of dataframe, Rows: {df.shape[0]}, Columns: {df.shape[1]}"
                )

                logger.info("Saving the raw dataframe into the artifacts folder..")
                self.config.feature_dir.mkdir(exist_ok=True)
                df.to_csv(self.config.feature_dir / "raw.csv", index=False)

                logger.info("Dropping unwanted columns..")
                df.drop(columns=self.config.drop_columns, inplace=True)

                logger.info("Train Test Split")
                train, test = train_test_split(
                    df,
                    test_size=self.config.INGESTION_PARAMS.test_size,
                    random_state=self.config.INGESTION_PARAMS.random_state,
                    stratify=df[self.config.target]
                    if self.config.INGESTION_PARAMS.stratify
                    else None,
                )

                logger.info("Saving the train and test into the artifacts folder..")
                self.config.ingested_dir.mkdir(exist_ok=True)
                train.to_csv(
                    self.config.ingested_dir / "train.csv", index=False, header=True
                )
                test.to_csv(
                    self.config.ingested_dir / "test.csv", index=False, header=True
                )

                mlflow.log_artifact(
                    str(self.config.ingested_dir / "train.csv"),
                    artifact_path="ingested_data",
                )

                mlflow.log_artifact(
                    str(self.config.ingested_dir / "test.csv"),
                    artifact_path="ingested_data",
                )

                mlflow.log_artifact(
                    str(self.config.feature_dir / "raw.csv"),
                    artifact_path="feature_data",
                )

                mlflow.log_metrics(
                    {
                        "test_size": self.config.INGESTION_PARAMS.test_size,
                        "random_state": self.config.INGESTION_PARAMS.random_state,
                        "stratify": self.config.INGESTION_PARAMS.stratify,
                        "target": self.config.target,
                        "database": self.config.database_name,
                        "collection": self.config.collection_name,
                    }
                )

                if not train.empty and not test.empty:
                    mlflow.set_tag("Status", "Pass")
                else:
                    mlflow.set_tag("Status", "Fail")

                return (train, test)

            except Exception as e:
                mlflow.set_tag("Status", "Fail")
                mlflow.log_text(traceback.format_exc(), "data_ingestion_traceback.txt")
                logger.exception("Exception occured while ingesting the data.")
                raise BankMarketingException(e, sys) from e
