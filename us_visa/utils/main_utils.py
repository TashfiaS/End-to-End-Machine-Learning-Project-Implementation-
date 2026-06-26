import os
import sys

import numpy as np
import dill
import yaml
from pandas import DataFrame
from sklearn.metrics import f1_score, precision_score, recall_score

from us_visa.exception import USvisaException
from us_visa.logger import logging


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise USvisaException(e, sys) from e


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise USvisaException(e, sys) from e


def load_object(file_path: str) -> object:
    logging.info("Entered the load_object method of utils")
    try:
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)
        logging.info("Exited the load_object method of utils")
        return obj
    except Exception as e:
        raise USvisaException(e, sys) from e


def save_numpy_array_data(file_path: str, array: np.array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise USvisaException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise USvisaException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    logging.info("Entered the save_object method of utils")
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info("Exited the save_object method of utils")
    except Exception as e:
        raise USvisaException(e, sys) from e


def drop_columns(df: DataFrame, cols: list) -> DataFrame:
    logging.info("Entered drop_columns method of utils")
    try:
        df = df.drop(columns=cols, axis=1)
        logging.info("Exited the drop_columns method of utils")
        return df
    except Exception as e:
        raise USvisaException(e, sys) from e


def get_classification_score(y_true, y_pred):
    from us_visa.entity.artifact_entity import ClassificationMetricArtifact
    try:
        model_f1_score = f1_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)
        model_precision_score = precision_score(y_true, y_pred)
        classification_metric = ClassificationMetricArtifact(
            f1_score=model_f1_score,
            precision_score=model_precision_score,
            recall_score=model_recall_score
        )
        return classification_metric
    except Exception as e:
        raise USvisaException(e, sys) from e
