from src.bank_marketing_ml.config.configuration import ConfigurationManager
from src.bank_marketing_ml.components.model_trainer import ModelTrainer

class ModelTrainerTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_trainer_config()
        model_trainer = ModelTrainer(
            config=model_trainer_config
        )

        model_trainer.initiate_trainer()