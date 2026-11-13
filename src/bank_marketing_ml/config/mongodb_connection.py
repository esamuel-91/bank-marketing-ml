import pymongo
from dotenv import load_dotenv
from from_root import from_root
import certifi
import sys

from src.bank_marketing_ml.exception import BankMarketingException
from src.bank_marketing_ml.utils.utils_main import get_env_var
from src.bank_marketing_ml import logger

ENV_DIR = from_root("atlas-credentials.env")
load_dotenv(ENV_DIR)
ca = certifi.where()


class MongoDBClient:
    """
    Makes a connection with the mongo client and returns the ping to check the connection status.
    Connects to specific database name

    Raises:
        ValueError: if the database name is not set
        BankMarketingException: if the connection to the database fails
    Return:
        None
    """

    client = None

    def __init__(self, database_name: str) -> None:
        self.database_name = database_name

        try:
            logger.info("Loading the URI key..")
            mongo_uri = get_env_var("MONGODB_URI")

            logger.info("Connecting to the client using URI key..")
            MongoDBClient.client = pymongo.MongoClient(mongo_uri, tlsCAFile=ca)

            logger.info("Checking the client connection using ping command.")
            response = MongoDBClient.client.admin.command("ping")

            if response.get("ok") != 1.0:
                raise ConnectionError("Error occured while connecting to Mongo Client.")

            if self.database_name == "":
                raise ValueError("Database name must not be empty!")

            self.client = MongoDBClient.client
            self.database = self.client[self.database_name]

            logger.info(
                f"Successfully connected to specific Mongo Database: {self.database_name}"
            )

        except Exception as e:
            logger.exception(
                "Exception occured while connecting to specific mongo database."
            )
            raise BankMarketingException(e, sys) from e
