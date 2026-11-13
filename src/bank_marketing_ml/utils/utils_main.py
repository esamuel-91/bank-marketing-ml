from src.bank_marketing_ml.exception import BankMarketingException
from src.bank_marketing_ml import logger
from box import ConfigBox
from ensure import ensure_annotations
from typing import Any
import joblib
import yaml
import json

import pandas as pd
import numpy as np

from pathlib import Path
import os
import sys


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    reads yaml file from the path provided and returns the output as ConfigBox type

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
        list_of_directories (list): directories to be created in a list
        verbose (bool):
            - if true:
                - logs the progress of the directory created
    """
    try:
        if len(list_of_directories) == 0:
            raise ValueError("List of directories provided is empty.")

        for directory in list_of_directories:
            os.makedirs(directory, exist_ok=True)

            if verbose:
                logger.info(f"Directory created: {directory}")
    except Exception as e:
        logger.exception("Exception occured while creating the directories.")
        raise BankMarketingException(e, sys) from e


@ensure_annotations
def save_json(path_to_json: Path, data: dict) -> None:
    """
    saves the data in json format to the path provided.

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
    loads the jsonfile from the path provided and returns the output as ConfigBox type

    Args:
        path_to_json (Path): path to the json file
    Return:
        ConfigBox: ConfigBox type
    """
    try:
        with open(file=path_to_json, mode="r") as jsonfile:
            data = json.load(jsonfile)
            logger.info(f"JSON file loaded successfully from: {path_to_json}")

        return ConfigBox(data)
    except Exception as e:
        logger.exception("Exception occured while loading the jsonfile.")
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
    loads the binary file from the path provided and returns it as output

    Args:
        path_to_bin (Path): path to the binary file
    Return:
        - binary file
    """
    try:
        bin = joblib.load(path_to_bin)
        logger.info(f"Binary file loaded successfully from: {path_to_bin}")
        return bin
    except Exception as e:
        logger.exception("Exception occured while loading the binary file.")
        raise BankMarketingException(e, sys) from e


@ensure_annotations
def save_arr(filepath: Path, data: np.ndarray):
    """
    saves the array to the path provided

    Args:
        filepath (Path): save path for the array file
        data (np.ndarray): data to be saved
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
    loads the array file from the path provided and returns the array file

    Args:
        filepath (Path): path to the saved array file
    Return:
        np.ndarray
    """
    try:
        with open(file=filepath, mode="rb") as file_obj:
            arr = np.load(file=file_obj)
            logger.info(f"array file loaded successfully from: {filepath}")

        return arr
    except Exception as e:
        logger.exception("Exception occured while loading the array file.")
        raise BankMarketingException(e, sys) from e


@ensure_annotations
def get_size(filepath: Path) -> str:
    """
    gets the size of the file from the path provided and returns the size in KB

    Args:
        filepath (Path): filepath for which you want to get the size
    Return:
        - size in kb
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
    Fetches the key from the env file and returns the value

    Args:
        key (str): key name for which you want to get the value
    Raises:
        EnvironmentError: if key is not set or the value of the key is empty
    Return:
        value of the key
    """
    try:
        value = os.getenv(key=key)

        if not value:
            raise EnvironmentError(f"Environment key: {key} not set or key is empty.")

        return value
    except Exception as e:
        logger.exception("Exception occured while getting the value for env key.")
        raise BankMarketingException(e, sys) from e


@ensure_annotations
def drop_columns(dataframe: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    drops the columns mentioned in the list from the specific dataframe and returns the rest of the dataframe

    Args:
        dataframe (pd.DataFrame): dataframe from which columns need to be dropped.
        columns(list): list of columns to be dropped
    Return:
        pd.DataFrame
    """
    try:
        data = dataframe.copy()
        data.drop(columns=columns, inplace=True)
        logger.info(f"Columns: {columns} dropped successfully from dataframe.")
        return data
    except Exception as e:
        logger.exception(
            "Exception occured while dropping the columns from the dataframe."
        )
        raise BankMarketingException(e, sys) from e
