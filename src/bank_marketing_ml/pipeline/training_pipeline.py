from src.bank_marketing_ml.exception import BankMarketingException
from src.bank_marketing_ml import logger

from src.bank_marketing_ml.pipeline.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline,
)

import argparse
import mlflow
import sys


class TrainingPipeline:
    def __init__(self) -> None:
        self.stages = {
            "data_ingestion": ("Data Ingestion Stage", DataIngestionTrainingPipeline)
        }

    def run_stage(self, stage_name: str):
        if stage_name not in self.stages:
            raise ValueError(f"Invalid stage name: {stage_name}")

        stage_title, pipeline_class = self.stages[stage_name]

        try:
            logger.info(f">>>>> {stage_title} started <<<<<")
            pipeline = pipeline_class()
            pipeline.main()
            logger.info(
                f">>>>> {stage_title} completed successfully <<<<<\nx=========x"
            )
        except Exception as e:
            logger.exception(f"Exception occured while running stage: {stage_name}")
            raise BankMarketingException(e, sys) from e

    def run_all(self):
        for stage_name in self.stages:
            self.run_stage(stage_name)


def main():
    parser = argparse.ArgumentParser(description="Bank Marketing ML Training Pipeline")

    parser.add_argument(
        'stage',
        nargs="?",
        default="all",
        choices=[
            "all",
            "data_ingestion",
            "data_validation",
            "data_transformation",
            "model_trainer",
            "model_evaluation",
            "model_pusher",
        ],
        help="Specify the pipeline stage to execute.",
    )

    args = parser.parse_args()

    try:
        mlflow.set_experiment("Bank Marketing")
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

        with mlflow.start_run(run_name=args.stage):
            pipeline = TrainingPipeline()

            if args.stage == "all":
                pipeline.run_all()
            else:
                pipeline.run_stage(args.stage)

    except Exception as e:
        logger.exception("Training pipeline execution failed.")
        raise BankMarketingException(e, sys) from e


if __name__ == "__main__":
    main()
