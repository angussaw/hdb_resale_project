"""Utils.py contains the general functions that will be used in during the end-to-end
 pipeline of hdb estimator
"""
from contextlib import contextmanager
import hashlib
from importlib import import_module
import logging
import logging.config
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Callable, List, Tuple, Iterable

import numpy as np
import glob
from omegaconf import OmegaConf
import pandas as pd
import yaml
from omegaconf import DictConfig
import json
import requests
from geopy.distance import geodesic

logger = logging.getLogger(__name__)


@contextmanager
def timer(task: str = "Task"):
    """Logs how much time a code block takes

    Args:
        task (str, optional): Name of task, for logging purposes. Defaults to "Task".

    Example:

        with timer("showing example"):
            examplefunction()
    """
    start_time = time.time()
    yield
    logger.info(f"{task} completed in {time.time() - start_time:.5} seconds ---")


def setup_logging(
    logging_config_path="./conf/logging.yaml", default_level=logging.INFO
):
    """Set up configuration for logging utilities.

    Args:
        logging_config_path (str, optional): Path to YAML file containing configuration for
                                             Python logger. Defaults to "./conf/base/logging.yml".
        default_level (_type_, optional): logging object. Defaults to logging.INFO.
    """
    try:
        with open(logging_config_path, "rt") as file:
            log_config = yaml.safe_load(file.read())
            logging.config.dictConfig(log_config)

    except Exception as error:
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=default_level,
        )
        logger.info(error)
        logger.info("Logging config file is not found. Basic config is being used.")


# def init_mlflow(mlflow_config: dict) -> Tuple[str, str]:
#     """initialises mlflow parameters - tracking URI and experiment name.

#     Takes in a configuration dictionary and sets the tracking URI
#     and MLFlow experiment name. Returns the artifact name and the
#     mlflow run description.

#     Args:
#         mlflow_config (dict): A dictionary containing the configurations
#             of the mlflow run.

#     Returns:
#         artifact_name (str): Name of the artifact which the resultant
#             trained model will be saved as. If none specified, the file
#             will be saved as a hashed datetime.

#         description (str): Description of the mlflow run, if any.
#     """

#     logger.info("Logging to MLFlow at %s", os.getenv("MLFLOW_TRACKING_URI"))

#     mlflow_experiment_name = mlflow_config["experiment_name"]
#     mlflow.set_experiment(mlflow_experiment_name)
#     logger.info("Logging to MLFlow Experiment: %s", mlflow_experiment_name)

#     if mlflow_config["artifact_name"]:
#         artifact_name = mlflow_config["artifact_name"]
#     else:
#         hashlib.sha1().update(str(time.time()).encode("utf-8"))
#         artifact_name = hashlib.sha1().hexdigest()[:15]
#     return artifact_name, mlflow_config.get("description", "")


def check_envvars() -> bool:
    """Checks for MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD envars.
    Addtionally, alerts the user if AWS credentials are set.
    """

    return_status = True

    for envvar in [
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
        "MLFLOW_TRACKING_URI",
    ]:
        try:
            os.environ[envvar]
        except KeyError:
            logger.warning("<< %s >> not found", envvar)
            return_status = False
            break

    if os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_SECRET_ACCESS_KEY"):
        logger.info("AWS Credentials found; ignore if this is intended.")

    return return_status


def read_data(
    data_path: str, concat: bool = True) -> pd.DataFrame:
    """Helper function to read data from a specified
    file path


    Args:
        data_path (str): file directory to read data files from
        concat (bool, optional): boolean to indicate whether to concat all data files. Defaults to True.

    Returns:
        pd.DataFrame: resulting dataframe read from file directory
    """

    if concat:
        all_files = glob.glob(os.path.join(data_path , "*.csv"))
        li = []
        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            li.append(df)

        dataframe = pd.concat(li, axis=0, ignore_index=True)

    else:
        dataframe = pd.read_csv(data_path, index_col=None, header=0)

    return dataframe
    
def construct_dated_filepath(original_path: str, date: str) -> str:
    """A helper function to append a date to the end of a filepath.
    The function takes in "/home/examplefile.csv" and "2023-03-04" and
    returns "/home/examplefile_2023-03-04.csv".

    Args:
        original_path (str): The original filepath
        date (str): A string of a date that is to be appended to the
            original filepath

    Returns:
        str: A filepath with the date appended at the end.
    """

    filename, file_ext = os.path.splitext(original_path)
    dated_path = f"{filename}_{date}{file_ext}"
    return dated_path


def generate_named_tmp_dir(dir_name: str) -> str:
    """_summary_

    Args:
        folder_name (str): Desired name of tmp folder

    Returns:
        str: tmp folder path
    """
    tmp_dir = tempfile.mkdtemp()
    named_tmp_dir = Path(os.sep.join([tmp_dir, dir_name]))
    named_tmp_dir.mkdir()

    return named_tmp_dir


def find_coordinates(add: str) -> tuple:
    """With the block number and street name, get the full address of the hdb flat,
    including the postal code, geogaphical coordinates (lat/long)

    Args:
        add (str): block number and street name

    Returns:
        tuple: latitude and longitude coordinates
    """
    
    # Do not need to change the URL
    url= "https://developers.onemap.sg/commonapi/search?returnGeom=Y&getAddrDetails=Y&pageNum=1&searchVal="+ add        
    
    # Retrieve information from website
    response = requests.get(url)
    try:
        data = json.loads(response.text) 
    except ValueError:
        print('JSONDecodeError')
        pass

    if len(data["results"]) != 0:

        result = data["results"][0]
        latitude, longitude = float(result["LATITUDE"]), float(result["LONGITUDE"])

    else:
        latitude, longitude = float("inf"), float("inf")

        return latitude, longitude


def find_nearest_amenities(flat_transaction,
                           amenity_details: pd.DataFrame,
                           radius: int,
                           period: bool,
                           coordinates_feature: str,
                           year_month_feature: str) -> tuple:
    """Function to find the number of amenities within radius
    of a flat, and also the flat's distance to the nearest amenity

    Args:
        flat_transaction (_type_): flat transaction details (eg year_month, coordinates)
        amenity_details (pd.DataFrame): amenity details (eg year_month, coordinates)
        radius (int): radius around the flat
        period (bool): whether to take into account the opening date of the amenity
        coordinates_feature(str): name of coordinates feature
        year_month_feature (str): name of year_month feature

    Returns:
        tuple: nearest amenities information for the flat
    """
    flat_coordinates = flat_transaction[coordinates_feature]
    transaction_year_month = flat_transaction[year_month_feature]
    no_of_amenities_within_radius = 0
    distance_to_nearest_amenity = float("inf")
    if flat_coordinates != (float("inf"), float("inf")):
        for ind, eachloc in enumerate(amenity_details.iloc[:,0]):
            amenity_coordinates = (amenity_details.iloc[ind,1],amenity_details.iloc[ind,2])
            if period:
                amenity_year_month = amenity_details.iloc[ind,3]
                if transaction_year_month >= amenity_year_month:
                    distance = float(str(geodesic(flat_coordinates,amenity_coordinates))[:-3])
            else:
                distance = float(str(geodesic(flat_coordinates,amenity_coordinates))[:-3])

            if distance <= radius:   # compute number of amenities in 2km radius
                no_of_amenities_within_radius += 1

            distance_to_nearest_amenity = min(distance, distance_to_nearest_amenity)
        
    return no_of_amenities_within_radius, distance_to_nearest_amenity