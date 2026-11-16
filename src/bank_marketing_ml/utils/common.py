from pathlib import Path
from src.bank_marketing_ml.exception import BankMarketingException
from src.bank_marketing_ml import logger
from box import ConfigBox
from ensure import ensure_annotations
from typing import Any
import joblib
import json
import yaml
import pandas as pd
import numpy as np

import os
import sys


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    reads the yaml file from the path provided and returns the output as ConfigBox type

    Args:
        path_to_yaml (Path): path to the yaml file
    Raises:
        ValueError: if the yaml file is empty.
    Return:
        ConfigBox: ConfigBox type
    """
    try:
        with open(file=path_to_yaml, mode="r") as yamlfile:
            content = yaml.safe_load(yamlfile)
            logger.info(f"YAML file loaded successfully from: {path_to_yaml}")

            if not content:
                raise ValueError("YAML file is empty.")
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
            if true:
                - logs the progess of directory created
    """
    try:
        if len(list_of_directories) == 0:
            raise ValueError("List must not be empty..")

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
            logger.info(f"Jsonfile saved successfully at: {path_to_json}")
    except Exception as e:
        logger.exception("Exception occured while saving the json file.")
        raise BankMarketingException(e, sys) from e


@ensure_annotations
def load_json(path_to_json: Path) -> ConfigBox:
    """
    loads the saved json from the path provided and return the output as ConfigBox type

    Args:
        path_to_json (Path): path to the json file
    Return:
        ConfigBox: ConfigBox type
    """
    try:
        with open(file=path_to_json, mode="r") as jsonfile:
            data = json.load(jsonfile)
            logger.info(f"Jsonfile loaded successfully from: {path_to_json}")
        return ConfigBox(data)
    except Exception as e:
        logger.exception("Exception occured while loading the json file.")
        raise BankMarketingException(e, sys) from e


def save_bin(path_to_bin: Path, data: Any):
    """
    saves the data in binary format to the path provided

    Args:
        path_to_bin (Path): save path for the binary file
        data (Any): data to be saved in binary format
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
    loads the binary file from the path provided and returns the output as binary

    Args:
        path_to_bin (Path): path to the binary file
    Return:
        - binary file
    """
    try:
        bin_file = joblib.load(filename=path_to_bin)
        logger.info(f"Binary file loaded successfully from: {path_to_bin}")

        return bin_file
    except Exception as e:
        logger.exception("Exception occured while loading the binary file.")
        raise BankMarketingException(e, sys) from e


@ensure_annotations
def save_arr(path_to_arr: Path, data: np.ndarray):
    """
    saves the array to the path provided

    Args:
        path_to_arr (Path): save path for the array file
        data (np.ndarray): data to be saved as array
    """
    try:
        with open(file=path_to_arr, mode="wb") as file_obj:
            np.save(file=file_obj, arr=data)
            logger.info(f"Array file saved successfully at: {path_to_arr}")
    except Exception as e:
        logger.exception("Exception occured while saving the array file.")
        raise BankMarketingException(e, sys) from e


@ensure_annotations
def load_arr(path_to_arr: Path) -> np.ndarray:
    """
    loads the array file from the path provided and returns it

    Args:
        path_to_arr (Path): path to the array file
    Return:
         - np.ndarray
    """
    try:
        with open(file=path_to_arr, mode="rb") as file_obj:
            arr = np.load(file=file_obj)
            logger.info(f"Array file loaded successfully from: {path_to_arr}")
        return arr
    except Exception as e:
        logger.exception("Exception occured while loading the array file.")
        raise BankMarketingException(e, sys) from e


@ensure_annotations
def drop_columns(dataframe: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    drops the columns from the dataframe passed and returns the remaining dataframe

    Args:
        dataframe (pd.DataFrame): dataframe from which the columns need to be dropped
        columns (list): list of columns to be dropped
    """
    try:
        data = dataframe.copy()
        data.drop(columns=columns, inplace=True)
        logger.info(f"Columns: {columns} successfully dropped..")
        return data
    except Exception as e:
        logger.exception(
            "Exception occured while dropping the columns from the dataframe."
        )
        raise BankMarketingException(e, sys) from e


@ensure_annotations
def get_size(filepath: Path) -> str:
    """
    gets the size of the file from the path provided and returns the size in KB

    Args:
        filepath (Path): path to the file
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
    Fetches the key from the env file and return the value of the key

    Args:
        key (str): key name for which you want to fetch the value
    Raises:
        EnvironmentError: if the key is not set of the value of the key is empty.
    Return:
        - value of the key
    """
    try:
        value = os.getenv(key=key)

        if not value:
            raise EnvironmentError(
                f"Environment Error key: {key} not set or the value of the key is empty."
            )
        return value
    except Exception as e:
        logger.exception("Exception occured while getting the value of ENV KEY.")
        raise BankMarketingException(e, sys) from e
