from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    feature_dir: Path
    ingested_data: Path
    database_name: str
    collection_name: str
    target: str
    INGESTION_PARAMS: dict


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    train_path: Path
    test_path: Path
    drift_path: Path
    STATUS_FILE: Path
    drift_share_threshold: float
    missing_rate: float
    VALIDATION_SCHEMA: dict


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    train_path: Path
    test_path: Path
    arr_path: Path
    objs_path: Path
    target: str
    TRANSFORMATION_PARAMS: dict


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_arr: Path
    model_path: Path
    MODEL_PARAMS: dict


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_arr: Path
    model_path: Path
    metrics_file_path: Path


@dataclass(frozen=True)
class ModelPusherConfig:
    root_dir: Path
    model_dir: Path
    model_name: str
    model_format: str
    export_dir: Path
    export_model_name: str
