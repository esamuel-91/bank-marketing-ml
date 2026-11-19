from src.bank_marketing_ml.exception import BankMarketingException
from src.bank_marketing_ml import logger

from src.bank_marketing_ml.utils.common import load_bin
import sys 

class PredictPipeline:
    def __init__(self) -> None:
        pass

    def predict(self):
        """
        Function is used to make prediction on customer input data.
        """
        try:
            pass
        except Exception as e:
            logger.exception("Prediction Failed...")
            raise BankMarketingException(e,sys)