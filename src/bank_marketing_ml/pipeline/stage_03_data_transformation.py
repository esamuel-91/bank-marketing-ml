from src.bank_marketing_ml.config.configuration import ConfigurationManager
from src.bank_marketing_ml.components.data_transformation import DataTransformation

class DataTransformationTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_transformation_config()
        data_transformation = DataTransformation(
            config=data_transformation_config
        )

        data_transformation.initiate_transformation()