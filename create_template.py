import os 
import logging 
from pathlib import Path
from from_root import from_root

os.makedirs('logs',exist_ok=True) 
LOG_FILE_PATH = os.path.join('logs','project_skeleton.log')
logging.basicConfig(
    filename= LOG_FILE_PATH,
    level=logging.INFO,
    format="[%(asctime)s]: %(message)s"
)

DIR_PATH = from_root('src/bank_marketing_ml')

list_of_files = [
    DIR_PATH / "__init__.py",
    DIR_PATH / "components/__init__.py",
    DIR_PATH / "components/data_ingestion.py",
    DIR_PATH / "components/data_validation.py",
    DIR_PATH / "components/data_transformation.py",
    DIR_PATH / "components/model_trainer.py",
    DIR_PATH / "components/model_evaluation.py",
    DIR_PATH / "components/model_pusher.py",
    DIR_PATH / "pipeline/__init__.py",
    DIR_PATH / "pipeline/training_pipeline.py",
    DIR_PATH / "pipeline/stage_01_data_ingestion.py",
    DIR_PATH / "pipeline/stage_02_data_validation.py",
    DIR_PATH / "pipeline/stage_03_data_transformation.py",
    DIR_PATH / "pipeline/stage_04_model_trainer.py",
    DIR_PATH / "pipeline/stage_05_model_evaluation.py",
    DIR_PATH / "pipeline/stage_06_model_pusher.py",
    DIR_PATH / "pipeline/prediction_pipeline.py",
    DIR_PATH / "utils/__init__.py",
    DIR_PATH / "utils/common.py",
    DIR_PATH / "entity/__init__.py",
    DIR_PATH / "entity/config_entity.py",
    DIR_PATH / "config/__init__.py",
    DIR_PATH / "config/configuration.py",
    DIR_PATH / "config/mongodb_connection.py",
    DIR_PATH / "config/aws_connection.py",
    DIR_PATH / "constants/__init__.py",
    DIR_PATH / "exception/__init__.py",
    "config/config.yaml",
    "config/params.yaml",
    "config/schema.yaml",
    "Dockerfile",
    "app.py",
    "requirements.txt",
    "requirements_dev.txt",
    "research/eda.ipynb",
    "research/trials.ipynb",
    "research/model_training.ipynb",
    "templates/form.html",
    "templates/index.html",
    "templates/result.html",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"Directory created: {filedir} for filename: {filename}")

    if (not filepath.exists()) or (os.path.getsize(filepath)==0):
        filepath.touch(exist_ok=True)
        logging.info(f"Created empty file: {filename}")

    else:
        logging.info(f"{filepath} already exists..")