from src.bank_marketing_ml.exception import BankMarketingException
from src.bank_marketing_ml import logger
from src.bank_marketing_ml.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

import mlflow
import sys

mlflow.set_experiment("Bank Marketing")
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_registry_uri("sqlite:///mlflow.db")

with mlflow.start_run():
    STAGE_NAME = "Data Ingestion Stage"

    try:
        logger.info(f">>>>> {STAGE_NAME} started <<<<<")
        data_ingestion = DataIngestionTrainingPipeline()
        data_ingestion.main()
        logger.info(f">>>>> {STAGE_NAME} completed <<<<<\nx========x")
    except Exception as e:
        logger.exception("Exception occured while initiating ingestion pipeline.")
        raise BankMarketingException(e, sys) from e
