from src.bank_marketing_ml.entity.config_entity import DataIngestionConfig
from src.bank_marketing_ml.entity.config_entity import DataValidationConfig
from src.bank_marketing_ml.entity.config_entity import DataTransformationConfig
from src.bank_marketing_ml.entity.config_entity import ModelTrainerConfig
from src.bank_marketing_ml.entity.config_entity import ModelEvaluationConfig
from src.bank_marketing_ml.entity.config_entity import ModelPusherConfig

from src.bank_marketing_ml.constants import *
from src.bank_marketing_ml.utils.common import read_yaml
from src.bank_marketing_ml.utils.common import create_directories

from pathlib import Path


class ConfigurationManager:
    def __init__(
        self,
        config_file_path=CONFIG_FILE_PATH,
        params_file_path=PARAMS_FILE_PATH,
        schema_file_path=SCHEMA_FILE_PATH,
    ):
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)
        self.schema = read_yaml(schema_file_path)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        params = self.params.INGESTION
        schema = self.schema

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            feature_dir=Path(config.feature_dir),
            ingested_dir=Path(config.ingested_dir),
            database_name=config.MONGO_KEYS.database_name,
            collection_name=config.MONGO_KEYS.collection_name,
            target=schema.TARGET_COLUMN.name,
            drop_columns=schema.DROP_COLUMN,
            INGESTION_PARAMS=params,
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=Path(config.root_dir),
            train_path=Path(config.train_path),
            test_path=Path(config.test_path),
            drift_path=Path(config.drift_path),
            STATUS_FILE=Path(config.STATUS_FILE),
            target=schema.TARGET_COLUMN.name,
            VALIDATION_SCHEMA=schema,
        )

        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        params = self.TRANSFORMATION
        schema = self.schema

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            train_path=Path(config.train_path),
            test_path=Path(config.test_path),
            preprocessor_name=config.preprocessor_name,
            target=schema.TARGET_COLUMN.name,
            TRANSFORMATION_PARAMS=params,
        )

        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.MODELS

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            train_arr=Path(config.train_arr),
            model_name=config.model_name,
            MODEL_PARAMS=params,
        )

        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        schema = self.schema.METRICS

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            test_arr=Path(config.test_arr),
            model_path=Path(config.model_path),
            metrics_file_path=Path(config.metrics_file_path),
            EVALUATION_SCHEMA=schema,
        )

        return model_evaluation_config

    def get_model_pusher_config(self) -> ModelPusherConfig:
        config = self.config.model_pusher

        create_directories([config.root_dir])

        model_pusher_config = ModelPusherConfig(
            root_dir=Path(config.root_dir),
            model_dir=Path(config.model_dir),
            model_name=config.model_name,
            model_format=config.model_format,
            export_dir=Path(config.export_dir),
            export_model_name=config.export_model_name,
        )

        return model_pusher_config
