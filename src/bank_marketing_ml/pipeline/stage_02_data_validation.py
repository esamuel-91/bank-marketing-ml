from src.bank_marketing_ml.config.configuration import ConfigurationManager
from src.bank_marketing_ml.components.data_validation import DataValidation

class DataValidationTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_validation_config()
        data_validation = DataValidation(
            config=data_validation_config
        )

        data_validation.initiate_validation()