import pymongo
from dotenv import load_dotenv
import certifi
from from_root import from_root

from src.bank_marketing_ml.exception import BankMarketingException
from src.bank_marketing_ml.utils.common import get_env_var
from src.bank_marketing_ml import logger

import sys

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

    def __init__(self, database_name: str = None):
        self.database_name = database_name
        try:
            if self.database_name is None:
                raise ValueError(f"Database: {self.database_name} must not be empty.")
            mongo_url = get_env_var("MONGODB_URI")
            MongoDBClient.client = pymongo.MongoClient(mongo_url, tlsCAFile=ca)
            MongoDBClient.client.admin.command("ping")

            self.client = MongoDBClient.client
            self.database = self.client[self.database_name]

            logger.info("Database successfully created/connected.")

        except Exception as e:
            logger.exception(
                "Exception occured while connecting to the specific mongo database."
            )
            raise BankMarketingException(e, sys) from e
