from src.bank_marketing_ml.exception import BankMarketingException
from src.bank_marketing_ml import logger
from box import ConfigBox
from ensure import ensure_annotations
from typing import Any
import joblib
import json
import yaml

from pathlib import Path
import os
import sys


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    reads the yaml file from the path provided and returns the output as ConfigBox type

    Args:
        path_to_yaml (Path): path to the yaml file
    Raises:
        ValueError: if the yaml file is empty
        FileNotFoundError: if path provided does not exist
        BankMarketingException: error occured while return the output as ConfigBox
    Return:
        ConfigBox: ConfigBox type
    """
    try:
        if not path_to_yaml.exists():
            raise FileNotFoundError(f"Path provided does not exist: {path_to_yaml}")
        with open(file=path_to_yaml, mode="r") as yamlfile:
            content = yaml.safe_load(yamlfile)
            if not content:
                raise ValueError("YAML file is empty.")
            return ConfigBox(content)
    except Exception as e:
        logger.exception("Exception occured while reading the yaml file.")
        raise BankMarketingException(e, sys) from e


@ensure_annotations
def create_directories(list_of_directories: list, verbose: bool = True):
    """
    create the directories mentioned in the list

    Args:
        list_of_directories (list): list of directories to be created
        verbose (bool):
            if true:
                - logs the status of each directory created
    """
    try:
        if len(list_of_directories) == 0:
            raise ValueError("List provided is empty.")
        for directory in list_of_directories:
            os.makedirs(directory, exist_ok=True)

            if verbose:
                logger.info(f"Directory created: {directory}")
    except Exception as e:
        logger.exception("Exception occured while creating the directories.")
        raise BankMarketingException(e, sys) from e


@ensure_annotations
def save_json(path_to_json: Path, data: dict):
    """
    saves the data in json format to the path provided

    Args:
        path_to_json (Path): save path for the json file
        data (dict): data to be saved in json format
    """
    try:
        with open(file=path_to_json, mode="w") as jsonfile:
            json.dump(data, jsonfile, indent=4)
            logger.info(f"Json file saved successfully at: {path_to_json}")
    except Exception as e:
        logger.exception("Exception occured while saving the json file.")
        raise BankMarketingException(e, sys) from e


@ensure_annotations
def load_json(path_to_json: Path) -> ConfigBox:
    """
    loads the json data from the path provided and returns the output as ConfigBox type

    Args:
        path_to_json (Path): path to the json file
    Return:
        ConfigBox: ConfigBox type
    """
    try:
        with open(file=path_to_json, mode="r") as jsonfile:
            data = json.load(jsonfile)
            logger.info(f"Json file loaded successfully from: {path_to_json}")
            return ConfigBox(data)
    except Exception as e:
        logger.exception("Exception occured while loading the json file.")
        raise BankMarketingException(e, sys) from e


def save_bin(path_to_bin: Path, data: Any):
    """
    saves the data in binary format to the path provided

    Args:
        path_to_bin (Path): save path of the binary file
        data (Any): data to be saved as binary
    """
    try:
        joblib.dump(value=data, filename=path_to_bin)
        logger.info(f"Binary file saved successfully at: {path_to_bin}")
    except Exception as e:
        logger.exception("Exception occured while saving the binary file.")
        raise BankMarketingException(e, sys) from e


@ensure_annotations
def load_bin(path_to_bin: Path):
    """
    loads the binary file from the path provided and returns it as output

    Args:
        path_to_bin (Path): path to the binary file
    Return:
        - binary file
    """
    try:
        bin = joblib.load(filename=path_to_bin)
        logger.info(f"Binary file loaded successfully from: {path_to_bin}")
        return bin
    except Exception as e:
        logger.exception("Exception occured while loading the binary file.")
        raise BankMarketingException(e, sys) from e


@ensure_annotations
def get_size(filepath: Path) -> str:
    """
    gets the size of the file in KB and returns it

    Args:
        filepath (Path): filepath for which you want to get the size
    Return:
         - size in KB
    """
    try:
        size_in_kb = round(os.path.getsize(filepath) / 1024)
        logger.info(f"~ {size_in_kb} KB")
        return f"~ {size_in_kb} KB"
    except Exception as e:
        logger.exception("Exception occured while getting the size of the file.")
        raise BankMarketingException(e, sys) from e


@ensure_annotations
def get_env_var(key: str) -> str:
    """
    Fetches the key from the env folder and returns it value

    Args:
        key (str): key for which you want to get the value
    Return:
        - value of the key
    """
    try:
        value = os.getenv(key)
        if not value:
            raise EnvironmentError(
                f"Environment key: {key} is not set or the value is empty."
            )
        return value
    except Exception as e:
        logger.exception("Exception occured while getting the key variable value.")
        raise BankMarketingException(e, sys) from e
