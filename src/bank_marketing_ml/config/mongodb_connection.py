import pymongo
from src.bank_marketing_ml.exception import BankMarketingException
from src.bank_marketing_ml import logger
from dotenv import load_dotenv
from from_root import from_root
import certifi

from src.bank_marketing_ml.utils.common import get_env_var
import sys


ENV_DIR = from_root("atlas-credentials.env")
load_dotenv(ENV_DIR)
ca = certifi.where()


class MongoDBClient:
    """
    Establishes a connection with the mongo client using MONGO URI.
    Checks the connection using ping command. If connection is successful then it connects to specific database.

    Raies:
        ConnectionError: if the ping command fails
        ValueError: if the database name is empty
        BankMarketingException: if error occured while connecting to specific database
    Return:
        - None
    """

    client = None

    def __init__(self, database_name: str) -> None:
        self.database_name = database_name
        try:
            logger.info("Fetching the URI KEY..")
            mongo_uri = get_env_var("MONGODB_URI")

            logger.info("Connecting to mongo client...")
            MongoDBClient.client = pymongo.MongoClient(mongo_uri, tlsCAFile=ca)

            logger.info("Checking connection status using ping command...")
            response = MongoDBClient.client.admin.command("ping")

            if response.get("ok") != 1.0:
                raise ConnectionError("Mongo Client connection failed.")

            if not self.database_name:
                raise ValueError("Database name must not be empty.")
            self.client = MongoDBClient.client
            self.database = self.client[self.database_name]

            logger.info(f"Successfully connected to databse: {self.database_name}")
        except Exception as e:
            logger.exception(
                "Exception occured while connecting to the specific database."
            )
            raise BankMarketingException(e, sys) from e
