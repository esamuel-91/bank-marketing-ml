from src.bank_marketing_ml.config.configuration import ConfigurationManager
from src.bank_marketing_ml.components.data_ingestion import DataIngestion

class DataIngestionTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_ingestion_config()
        data_ingestion = DataIngestion(
            config=data_ingestion_config
        )

        data_ingestion.initiate_ingestion()