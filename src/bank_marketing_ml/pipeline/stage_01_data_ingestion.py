from src.bank_marketing_ml.config.configuration import ConfigurationManager
from src.bank_marketing_ml.components.data_ingestion import Dataingestion

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = Dataingestion(
            config=data_ingestion_config
        )

        data_ingestion.initiate_ingestion()