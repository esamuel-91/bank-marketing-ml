from src.bank_marketing_ml.config.configuration import DataIngestionConfig
from src.bank_marketing_ml.exception import BankMarketingException
from src.bank_marketing_ml import logger
from src.bank_marketing_ml.config.mongodb_connection import MongoDBClient
from src.bank_marketing_ml.utils.common import drop_columns

from sklearn.model_selection import train_test_split

import mlflow
import traceback

from pandas import DataFrame
import sys


class DataIngestion:
    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config

    def gather_collection_data(self):
        """
        Connects to the specific database and fetches a specific collection.
        Collection data is stored in list format

        Raises:
            ValueError: if the collection is empty / not set
            BankMarketingException: if error occured while getting the collection document as list

        Return:
            - data (list): collection data in list format
        """
        try:
            logger.info(f"Connecting to specific database through mongo client")
            client = MongoDBClient(database_name=self.config.database_name)

            if not self.config.collection_name:
                raise ValueError("Collection name must not be empty..")

            collection = client.database[self.config.collection_name]
            data = list(collection.find())

            if not data:
                raise ValueError(
                    f"Specific collection: {self.config.collection_name} is empty.."
                )

            return data
        except Exception as e:
            logger.exception(
                "Exception occured while fetching the collection data as list."
            )
            raise BankMarketingException(e, sys) from e

    def initiate_ingestion(self):
        """
        Loads the collected collection data as list.
        Creates a DataFrame from the collected data and splits the data into train and test.
        Saves the resulting artifacts into the artifacts folder.

        Uses customer logger and MLflow to log the artifacts and progress

        Return:
            tupe(DataFrame,DataFrame): train and test dataset as tuple
        """
        try:
            logger.info(f"Loading the collected collection data...")
            data = self.gather_collection_data()

            logger.info(f"Creating DataFrame from the collected data..")
            df = DataFrame(data)
            logger.info(
                f"Shape of the raw dataframe; Row: {df.shape[0]}, Columns: {df.shape[1]}"
            )
            logger.info("Saving the raw dataframe into the artifacts folder..")
            self.config.feature_dir.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.config.feature_dir, index=False)

            logger.info("Dropping the unwanted columns from the dataframe..")
            df = drop_columns(df, self.config.drop_columns)

            logger.info(f"Splitting the data into train and test")
            train, test = train_test_split(
                df,
                test_size=self.config.INGESTION_PARAMS.test_size,
                random_state=self.config.INGESTION_PARAMS.random_state,
                stratify=df[self.config.target]
                if self.config.INGESTION_PARAMS.stratify
                else None,
            )

            logger.info(
                f"Shape of the train dataframe; Row: {train.shape[0]}, Columns: {train.shape[1]}"
            )
            logger.info(
                f"Shape of the test dataframe; Row: {test.shape[0]}, Columns: {test.shape[1]}"
            )

            logger.info(
                "Saving the train and test artifacts into the artifacts folder.."
            )
            self.config.ingested_dir.mkdir(parents=True, exist_ok=True)
            train.to_csv(
                self.config.ingested_dir / "train.csv", index=False, header=True
            )
            test.to_csv(self.config.ingested_dir / "test.csv", index=False, header=True)

            mlflow.log_artifact(
                str(self.config.ingested_dir / "train.csv"),
                artifact_path="ingested_data",
            )
            mlflow.log_artifact(
                str(self.config.ingested_dir / "test.csv"),
                artifact_path="ingested_data",
            )
            mlflow.log_artifact(
                str(self.config.feature_dir), artifact_path="feature_data"
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
                "Pass" if not train.empty or not test.empty else "False",
            )

            return train, test
        except Exception as e:
            mlflow.log_text(traceback.format_exc(), "data_ingestion_traceback.txt")
            logger.exception("Exception occured while ingesting the data.")
            raise BankMarketingException(e, sys) from e
