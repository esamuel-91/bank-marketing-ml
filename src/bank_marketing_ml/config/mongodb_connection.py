from src.bank_marketing_ml.exception import BankMarketingException
from src.bank_marketing_ml import logger
import pymongo
from dotenv import load_dotenv
import certifi

from from_root import from_root
import sys
from src.bank_marketing_ml.utils.utils_main import get_env_var

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
            logger.info("Fetching the URI key..")
            mongo_uri = get_env_var("MONGODB_URI")
            MongoDBClient.client = pymongo.MongoClient(mongo_uri, tlsCAFile=ca)

            logger.info("Checking the client connection using the ping command..")
            response = MongoDBClient.client.admin.command("ping")

            if response.get("ok") != 1.0:
                raise ConnectionError(
                    "Connection Failed: Unable to connect to the client."
                )
            if self.database_name == "":
                raise ValueError("Database name must not be empty..")
            self.client = MongoDBClient.client
            self.database = self.client[self.database_name]

            logger.info(f"Successfully connected to the database: {self.database_name}")
        except Exception as e:
            logger.exception("Exception occured while conneting to the database.")
            raise BankMarketingException(e, sys) from e
