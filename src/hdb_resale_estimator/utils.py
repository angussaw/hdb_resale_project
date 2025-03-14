"""Utils.py contains the general functions that will be used in during the end-to-end
 pipeline of hdb estimator
"""
from contextlib import contextmanager
from geopy.distance import geodesic
import glob
import hashlib
import joblib
import json
import logging
import logging.config
import mlflow
import numpy as np
import os
import pandas as pd
from pathlib import Path
import requests
import sqlalchemy
import tempfile
import time
from typing import Tuple
import yaml

from hdb_resale_estimator.modeling.builder import ClassicalModelBuilder

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


def init_mlflow(mlflow_config: dict) -> Tuple[str, str]:
    """initialises mlflow parameters - tracking URI and experiment name.

    Takes in a configuration dictionary and sets the tracking URI
    and MLFlow experiment name. Returns the artifact name, the
    mlflow run description and the experiment ID.

    Args:
        mlflow_config (dict): A dictionary containing the configurations
            of the mlflow run.

    Returns:
        artifact_name (str): Name of the artifact which the resultant
            trained model will be saved as. If none specified, the file
            will be saved as a hashed datetime.

        description (str): Description of the mlflow run, if any.

        experiment_id (str): ID of the MLflow experiment that was set.
    """

    logger.info("Logging to MLFlow at %s", os.getenv("MLFLOW_TRACKING_URI"))

    mlflow_experiment_name = mlflow_config["experiment_name"]
    mlflow_experiment = mlflow.set_experiment(mlflow_experiment_name)
    mlflow_experiment_id = mlflow_experiment.experiment_id
    logger.info(f"Logging to MLFlow Experiment Name: {mlflow_experiment_name}, ID: {mlflow_experiment_id}")

    if mlflow_config["artifact_name"]:
        artifact_name = mlflow_config["artifact_name"]
    else:
        hashlib.sha1().update(str(time.time()).encode("utf-8"))
        artifact_name = hashlib.sha1().hexdigest()[:15]
    return artifact_name, mlflow_config.get("description", ""), mlflow_experiment_id


def read_data(source: str, params: dict) -> pd.DataFrame:
    """Helper function to read data either from csv or postgres

    Args:
        source (str): data source to read from
        params (dict): configuration parameters used to read data

    Returns:
        pd.DataFrame: Dataframe read from csv or postgres
    """

    if source == "csv":
        dataframe = read_csv(**params)

    elif source == "postgres":
        dataframe = extract_data_from_psql(**params)

    return dataframe


def read_csv(data_path: str, concat: bool = True) -> pd.DataFrame:
    """Helper function to read csv data from a specified
    file path

    Args:
        data_path (str): file directory to read data files from
        concat (bool, optional): boolean to indicate whether to concat all data files.
        Defaults to True.

    Returns:
        pd.DataFrame: resulting dataframe read from file directory
    """

    if concat:
        all_files = glob.glob(os.path.join(data_path, "*.csv"))
        li = []
        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            li.append(df)

        dataframe = pd.concat(li, axis=0, ignore_index=True)

    else:
        dataframe = pd.read_csv(data_path, index_col=None, header=0)

    return dataframe


def find_coordinates(add: str) -> tuple:
    """With the block number and street name, get the full address of the hdb flat,
    including the postal code, geogaphical coordinates (lat/long)

    Args:
        add (str): block number and street name

    Returns:
        tuple: latitude and longitude coordinates
    """
    # Do not need to change the URL
    url = f"https://www.onemap.gov.sg/api/common/elastic/search?searchVal={add}&returnGeom=Y&getAddrDetails=Y&pageNum=1"

    # Retrieve information from website
    response = requests.get(url)
    try:
        data = json.loads(response.text)
    except ValueError:
        print("JSONDecodeError")
        pass

    if len(data["results"]) != 0:
        result = data["results"][0]
        latitude, longitude = float(result["LATITUDE"]), float(result["LONGITUDE"])

    else:
        latitude, longitude = float("inf"), float("inf")

    return latitude, longitude


def calculate_haversine_distance(lat1: float,
                                lon1: float,
                                lat2: float,
                                lon2: float,
                                to_radians: bool=True,
                                earth_radius: int=6371) -> float:
    """Helper function to caluclate the haversine distance between two coordinates

    Args:
        lat1 (float): latitude of point 1
        lon1 (float): longtitude of point 1
        lat2 (float): latitude of point 2
        lon2 (float): longtitude of point 2
        to_radians (bool, optional): Converts coordinates to radians. Defaults to True.
        earth_radius (int, optional): Radius of the earth. Defaults to 6371.

    Returns:
        float: haversine distance between two coordinates
    """

    if to_radians:
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    a = np.sin((lat2-lat1)/2.0)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2

    return earth_radius * 2 * np.arcsin(np.sqrt(a))

def find_nearest_amenities(
    flat_transaction: pd.Series,
    amenity_details: pd.DataFrame,
    radius: int,
    period: bool,
    latitude_feature: str,
    longitude_feature: str,
    year_month_feature: str,
    return_nearest_amenity: bool = False,
) -> tuple:
    """Function to find the number of amenities within radius
    of a flat, and also the flat's distance to the nearest amenity

    Args:
        flat_transaction (pd.Series): flat transaction details (eg year_month, coordinates)
        amenity_details (pd.DataFrame): amenity details (eg year_month, coordinates)
        radius (int): radius around the flat
        period (bool): whether to take into account the opening date of the amenity
        coordinates_feature(str): name of coordinates feature
        year_month_feature (str): name of year_month feature
        return_nearest_amenity (bool, optional): whether to return the location of the nearest amenity.
                                                 Defaults to False.

    Returns:
        tuple: nearest amenities information for the flat
    """

    flat_coordinates = (
        flat_transaction[latitude_feature],
        flat_transaction[longitude_feature],
    )
    transaction_year_month = flat_transaction[year_month_feature]

    amenity_df_copy = amenity_details.copy()
    if flat_coordinates != (float("inf"), float("inf")):

        amenity_df_copy["FLAT_LATITUDE"] = flat_transaction[latitude_feature]
        amenity_df_copy["FLAT_LONGITUDE"] =  flat_transaction[longitude_feature]
        amenity_df_copy["distance_to_flat"] = calculate_haversine_distance(amenity_df_copy["LATITUDE"],
                                                                            amenity_df_copy["LONGITUDE"],
                                                                            amenity_df_copy["FLAT_LATITUDE"],
                                                                            amenity_df_copy["FLAT_LONGITUDE"])
        if period:
            amenity_df_copy = amenity_df_copy[amenity_df_copy["year_month"] <= transaction_year_month]
        
        no_of_amenities_within_radius = len(amenity_df_copy[amenity_df_copy["distance_to_flat"] <= radius])
        distance_to_nearest_amenity = min(amenity_df_copy["distance_to_flat"])
        nearest_amenity_name = amenity_df_copy[amenity_df_copy["distance_to_flat"] == distance_to_nearest_amenity]["address"].values[0]
        nearest_amenity_coordinates = (amenity_df_copy[amenity_df_copy["address"] == nearest_amenity_name]["LATITUDE"].values[0],
                                       amenity_df_copy[amenity_df_copy["address"] == nearest_amenity_name]["LONGITUDE"].values[0])
        
    else:
        no_of_amenities_within_radius = None
        distance_to_nearest_amenity = None
        nearest_amenity_coordinates = None
        nearest_amenity_name = None

    if return_nearest_amenity:
        return (
            no_of_amenities_within_radius,
            distance_to_nearest_amenity,
            nearest_amenity_coordinates,
            nearest_amenity_name,
        )
    else:
        return no_of_amenities_within_radius, distance_to_nearest_amenity


def check_postgres_env() -> None:
    """Verify that required Postgres environment variables have been exported.

    Raises:
        KeyError: Error is raised when env variable is not exported yet.
    """
    for envvar in [
        "POSTGRES_USER",
        "POSTGRES_PWD",
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
    ]:
        try:
            os.environ[envvar]
        except KeyError:
            logger.warning("<< %s >> not found", envvar)
            raise KeyError("Ensure that Postgres env is set before saving to Postgres")


def create_postgres_engine() -> sqlalchemy.engine:
    """Create engine to connect to Postgres database

    Raises:
        KeyError: Raise error if environment variable cannot be found

    Returns:
        sqlalchemy.engine: sqlalchemy engine for postgres database
    """
    check_postgres_env()

    url_object = sqlalchemy.engine.URL.create(
        "postgresql+psycopg2",
        username=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PWD"),
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        database=os.getenv("POSTGRES_DB"),
    )
    engine = sqlalchemy.create_engine(url_object, pool_size=5, pool_recycle=3600)

    return engine


def extract_data_from_psql(table_name: str, columns: list) -> pd.DataFrame:
    """Helper function to extract data from postgres database

    Args:
        table_name (str): name of the postgres table to extract data from
        columns (list): list of columns to extract from the postgres table

    Returns:
        pd.DataFrame: resulting dataframe extracted from postgres table
    """

    check_postgres_env()
    db_engine = create_postgres_engine()
    columns_query = ", ".join(['"' + column + '"' for column in columns])
    sql_query = sqlalchemy.text(f"""
        SELECT {columns_query} FROM {table_name}
        WHERE date_context = (SELECT MAX(date_context) FROM {table_name})
    """)
    # Extract data from postgres table
    with db_engine.begin() as conn:
        extracted_df = pd.read_sql(sql_query, conn)

    df_obj = extracted_df.select_dtypes("object")
    extracted_df[df_obj.columns] = df_obj.astype(str).apply(lambda x: x.str.rstrip())

    return extracted_df


def push_data_to_sql(
    db_engine: sqlalchemy.engine, data: pd.DataFrame, table_name: str
) -> None:
    """Save data into postgres database

    Args:
        db_engine (sqlalchemy.engine): Postgres database engine
        data (pd.DataFrame): Data that is to be saved
        table_name (str): Name of postgres table
    """

    with db_engine.begin() as conn:
        check_duplicate_date_input(conn, data, table_name)
        data.to_sql(
            table_name,
            conn,
            if_exists="append",
            index=False,
            chunksize=500,
            method="multi",
        )


def check_duplicate_date_input(
    conn: sqlalchemy.engine.base.Connection, data: pd.DataFrame, table_name: str
) -> None:
    """Check if date_context of data exist within the table to be appended

    Args:
        conn (sqlalchemy.engine.base.Connection): Connection to sqlalchemy
        data (pd.DataFrame): Data to be pushed to psql
        table_name (str): Table_name in psql server

    Raises:
        ValueError: Error is raised when date_context in table_name exists to prevent duplicate entries
    """

    # Check data for date related column
    if "date_context" in data.columns:
        reference_column = "date_context"
    else:
        raise ValueError(
            f"date_context or date_of_inference does not exist. Columns in data: {data.columns}"
        )

    # Get date_context or date_of_inference in string format from data
    reference_column_value = str(np.datetime64(data[reference_column].unique()[0], "D"))

    query = sqlalchemy.text(f"""SELECT COUNT(*) FROM {table_name} WHERE {reference_column} = '{reference_column_value}'""")

    result = conn.execute(query)
    for row in result:
        # row is a tuple for this query
        date_context_rows = row[0]
    if date_context_rows > 0:
        raise ValueError(
            f"{reference_column} {reference_column_value} already exist in {table_name}"
        )


def retrieve_builder(
    experiment_id: str, run_id: str, destination_path: str = "models"
) -> ClassicalModelBuilder:
    """
    Function to retrieve a trained model from MLFLow for inference

    Args:
        experiment_id (str): MLFlow experiment id
        run_id (str): MLFlow run id
        destination_path (str): Path to save model to. Defaults to "models"

    Returns:
        ClassicalModelBuilder: Builder object with trained model
    """

    artifact_uri = f"file:///mlflow/experiments/{experiment_id}/{run_id}/artifacts/model"
    logger.info("Downloading artifacts from MLFlow artifact URI: %s...", artifact_uri)
    try:
        mlflow.artifacts.download_artifacts(
            artifact_uri=artifact_uri, dst_path=destination_path
        )
    except Exception as mlflow_error:
        logger.exception("Failed to load artifact: %s", mlflow_error)
        raise mlflow_error

    logger.info("Artifact download successful")

    model_path = glob.glob(f"{destination_path}/model/*.joblib")[-1]
    builder = joblib.load(model_path)

    return builder

def generate_named_tmp_dir(dir_name: str) -> str:
    """Generate a temporary directory for storage of artifact to log into MLFLOW

    Args:
        dir_name (str): Name of temporary directory

    Returns:
        (str): temporary directory path
    """
    tmp_dir = tempfile.mkdtemp()
    named_tmp_dir = Path(os.sep.join([tmp_dir, dir_name]))
    named_tmp_dir.mkdir()

    return named_tmp_dir
