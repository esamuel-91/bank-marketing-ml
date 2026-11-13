import os 
import logging
from pathlib import Path

os.makedirs('logs',exist_ok=True)
LOG_FILE = os.path.join('logs','project_skeleton.log')
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="[%(asctime)s]: %(message)s"
)

project_name = 'bank_marketing_ml'

list_of_files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_validation.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_evaluation.py",
    f"src/{project_name}/components/model_pusher.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/prediction_pipeline.py",
    f"src/{project_name}/pipeline/training_pipeline.py",
    f"src/{project_name}/pipeline/stage_01_data_ingestion.py",
    f"src/{project_name}/pipeline/stage_02_data_validation.py",
    f"src/{project_name}/pipeline/stage_03_data_transformation.py",
    f"src/{project_name}/pipeline/stage_04_model_trainer.py",
    f"src/{project_name}/pipeline/stage_05_model_evaluation.py",
    f"src/{project_name}/pipeline/stage_06_model_pusher.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/utils_main.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/constants/__init__.py",
    f"src/{project_name}/exception/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/config/aws_connection.py",
    f"src/{project_name}/config/mongodb_connection.py",
    ".github/workflows/.gitkeep",
    "config/config.yaml",
    "config/params.yaml",
    "config/schema.yaml",
    "Dockerfile",
    "app.py",
    "test.py",
    "requirements_dev.txt",
    "requirements.txt",
    "research/eda.ipynb",
    "research/trials.ipynb",
    "templates/form.html",
    "templates/index.html",
    "templates/result.html",
    "dvc.yaml"

]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"Directory created: {filedir} for file: {filename}")

    if (not filepath.exists()) or (os.path.getsize(filepath)==0):
        filepath.touch(exist_ok=True)
        logging.info(f"Created an empty file: {filename}")
    else:
        logging.info(f"{filepath} already exists!")