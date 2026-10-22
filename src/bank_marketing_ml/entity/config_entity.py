from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    feature_dir: Path
    ingested_dir: Path
    database_name: str
    collection_name: str
    target: str
    drop_columns: str
    INGESTION_PARAMS: dict


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    train_path: Path
    test_path: Path
    drift_path: Path
    STATUS_FILE: Path
    target: str
    VALIDATION_SCHEMA: dict


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    train_path: Path
    test_path: Path
    preprocessor_name: str
    target: str
    TRANSFORMATION_PARAMS: dict


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_arr: Path
    model_name: Path
    MODEL_PARAMS: dict


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_arr: Path
    model_path: Path
    metrics_file_path: Path
    EVALUATION_SCHEMA: dict


@dataclass(frozen=True)
class ModelPusherConfig:
    root_dir: Path
    model_dir: Path
    model_name: str
    model_format: str
    export_dir: Path
    export_model_name: str
