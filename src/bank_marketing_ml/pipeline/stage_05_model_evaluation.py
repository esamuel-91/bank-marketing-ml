from src.bank_marketing_ml.config.configuration import ConfigurationManager
from src.bank_marketing_ml.components.model_evaluation import ModelEvaluation

class ModelEvaluationTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_evaluation_config()
        model_evaluation = ModelEvaluation(
            config=model_evaluation_config
        )

        model_evaluation.initiate_evaluation()