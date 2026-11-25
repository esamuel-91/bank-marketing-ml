import pymongo
from dotenv import load_dotenv
from from_root import from_root
import certifi
import sys

from src.bank_marketing_ml.exception import BankMarketingException
from src.bank_marketing_ml import logger
from src.bank_marketing_ml.utils.common import get_env_var

ENV_DIR = from_root("atlas-credentials.env")
load_dotenv(ENV_DIR)
ca = certifi.where()


class MongoDBClient:
    """
    Establishes a connection with the mongo client using MONGO URI.
    Checks the connection status using ping command.
        - if connection successful then connects to the specific database.

    Raise:
        ValueError: if the database name is not set
        ConnectionError: if the connection to the client is failed
        BankMarketingException: if error occured while connecting to specific database
    Return:
        None
    """

    client = None

    def __init__(self, database_name: str) -> None:
        self.database_name = database_name

        try:
            logger.info(f"Fetching URI key...")
            mongo_uri = get_env_var("MONGODB_URI")

            logger.info(f"Establishing the connection with the mongo client.")
            MongoDBClient.client = pymongo.MongoClient(mongo_uri, tlsCAFile=ca)

            logger.info(f"Checking the connection to the mongo client..")
            response = MongoDBClient.client.admin.command("ping")

            if response.get("ok") != 1.0:
                raise ConnectionError(f"Connection with the mongo client: Failed")

            if self.database_name is None:
                raise ValueError(f"Database name must not be empty..")
            else:
                self.client = MongoDBClient.client
                self.database = self.client[self.database_name]

                logger.info(
                    f"Successfully connected to the specific database: {self.database_name}"
                )

        except Exception as e:
            logger.exception(
                f"Exception occured while connecting to specific database: {self.database_name}"
            )
            raise BankMarketingException(e, sys) from e
