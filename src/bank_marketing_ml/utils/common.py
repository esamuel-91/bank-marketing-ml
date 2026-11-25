from src.bank_marketing_ml.exception import BankMarketingException
from src.bank_marketing_ml import logger
from box import ConfigBox
from typing import Any
from ensure import ensure_annotations
import json
import joblib
import yaml

from pandas import DataFrame
import numpy as np

from pathlib import Path
import sys
import os


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    reads the yaml file from the path provided and returns the output as ConfigBox type

    Args:
        path_to_yaml (Path): path to the yaml file
    Raises:
        ValueError: if yaml file is empty
    Return:
        ConfigBox: ConfigBox type
    """
    try:
        with open(file=path_to_yaml, mode="r") as yamlfile:
            content = yaml.safe_load(yamlfile)
            logger.info(f"YAML file loaded successfully from: {path_to_yaml}")

            if not content:
                raise ValueError("YAML file is empty..")

        return ConfigBox(content)
    except Exception as e:
        logger.exception("Exception occured while reading the yaml file.")
        raise BankMarketingException(e, sys) from e


@ensure_annotations
def create_directories(list_of_directories: list, verbose: bool = True):
    """
    creates the directories mentioned in the list

    Args:
        list_of_directories (list): list of directories to be created
        verbose (bool):
            - if true:
                - logs the progress of directory created
    """
    try:
        if len(list_of_directories) == 0:
            raise ValueError("Directory list is empty..")

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
        path_to_json (Path): save location of the json file
        data (dict): data to be saved in json format
    """
    try:
        with open(file=path_to_json, mode="w") as jsonfile:
            json.dump(data, jsonfile, indent=4)
            logger.info(f"JSON file saved successfully at: {path_to_json}")
    except Exception as e:
        logger.exception("Exception occured while saving the json file.")
        raise BankMarketingException(e, sys) from e


@ensure_annotations
def load_json(path_to_json: Path) -> ConfigBox:
    """
    loads the json file from the path provided and returns the output as ConfigBox type

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


@ensure_annotations
def save_arr(filepath: Path, data: np.ndarray):
    """
    saves the data in array format to the path provided

    Args:
        filepath (Path): save location of the array file
        data (np.ndarray): data to be saved as array
    """
    try:
        with open(file=filepath, mode="wb") as file_obj:
            np.save(file=file_obj, arr=data)

            logger.info(f"Array file saved successfully at: {filepath}")
    except Exception as e:
        logger.exception("Exception occured while saving the array file.")
        raise BankMarketingException(e, sys) from e


@ensure_annotations
def load_arr(filepath: Path) -> np.ndarray:
    """
    loads the array file from the path provided and returns the output as arr

    Args:
        filepath (Path): path to the array file
    Return:
        np.ndarray
    """
    try:
        with open(file=filepath, mode="rb") as file_obj:
            arr = np.load(file=file_obj)
            logger.info(f"Array file loaded successfully from: {filepath}")
            return arr
    except Exception as e:
        logger.exception("Exception occured while loading the array file.")
        raise BankMarketingException(e, sys) from e


@ensure_annotations
def get_size(filepath: Path) -> str:
    """
    gets the size of the file in KB from the path provided and returns it

    Args:
        filepath (Path): path of the file to get the size
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
def drop_columns(dataframe: DataFrame, columns: list) -> DataFrame:
    """
    Drops the columns mentioned in the list from the specific DataFrame and returns a new DataFrame

    Args:
        dataframe (DataFrame): dataframe from which the columns need to be dropped
        columns (list): list of columns to be dropped
    Return:
        - DataFrame
    """
    try:
        data = dataframe.copy()
        data.drop(columns=columns, inplace=True)
        logger.info(f"Columns: {columns} successfully dropped from the dataframe.")
        return data
    except Exception as e:
        logger.exception(
            "Exception occured while dropping the columns from the dataframe."
        )
        raise BankMarketingException(e, sys) from e


def save_bin(path_to_bin: Path, data: Any):
    """
    saves the data in binary format to the path provided

    Args:
        path_to_bin (Path): save location of the binary file
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
    loads the binary file from the path provided and returns the binary file.

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
def get_env_var(key: str) -> str:
    """
    Fetches the env key and return the value of the key

    Args:
        key (str): name of the env key
    Raises:
        EnvironmentError: if key is not set or empty
    Return:
        - value of the key
    """
    try:
        value = os.getenv(key=key)

        if not value:
            raise EnvironmentError(f"Envirnoment key: {key} not set or the key is empty.")

        return value
    except Exception as e:
        logger.exception("Exception occured while getting the value of env key.")
        raise BankMarketingException(e,sys) from e