"""
feature_engineering.py will contain the neccessary class to perform feature engineering.
"""
from dateutil.relativedelta import relativedelta
import functools
import logging
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm_pandas, tqdm

import hdb_resale_estimator as hdb_est

logger = logging.getLogger("__name__")
tqdm_pandas(tqdm())

class FeatureEngineer:
    """FeatureEngineer 
    """

    def __init__(self, params: dict) -> None:
        self.params = params

    def engineer_features(self, clean_hdb_data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from cleaned mppa data

        Returns:
           pd.DataFrame: Output dataframe containing
           each customer and their respective mppa derived features
        """
        clean_hdb_data = clean_hdb_data.head(50)

        logger.info("Generating lat long coordinates...")
        hdb_data = self.generate_long_lat(hdb_data = clean_hdb_data)

        logger.info("Getting nearest parks...")
        nearest_parks = self.get_nearests_amenities(hdb_data = hdb_data, amenity="parks")

        logger.info("Getting nearest schools...")
        nearest_schools = self.get_nearests_amenities(hdb_data = hdb_data, amenity="schools")

        logger.info("Getting nearest malls...")
        nearest_malls = self.get_nearests_amenities(hdb_data = hdb_data, amenity="malls")

        logger.info("Getting nearest MRT stations...")
        nearest_mrt_stations = self.get_nearests_amenities(hdb_data = hdb_data, amenity="MRT_stations", period=True)

        logger.info("Merging hdb derived features...")
        derived_features_hdb = pd.concat([hdb_data,
                                          nearest_parks,
                                          nearest_schools,
                                          nearest_malls,
                                          nearest_mrt_stations], axis = 1)

        return derived_features_hdb


    def generate_long_lat(self, hdb_data: pd.DataFrame):
        """_summary_

        Args:
            hdb_data (_type_): _description_
        """
        hdb_data["coordinates"] = hdb_data.progress_apply(lambda x: hdb_est.utils.find_postal(x["block"] + " " + x["street_name"]), axis = 1)

        return hdb_data
    

    def get_nearests_amenities(self, hdb_data: pd.DataFrame, amenity: str, period = False):
        """_summary_

        Args:
            hdb_data (_type_): _description_
        """
        amenity_coordinates_file_path = self.params[amenity]["coordinates"]
        amenity_coordinates = hdb_est.utils.read_data(data_path = amenity_coordinates_file_path, concat=False)
        if period:
            amenity_coordinates = amenity_coordinates.rename(columns={"Opening year": "YEAR", "Opening month": "MONTH"})
            amenity_coordinates["Opening month"] = pd.to_datetime(amenity_coordinates[['YEAR', 'MONTH']].assign(DAY=1))
            amenity_coordinates = amenity_coordinates[["Name","LATITUDE", "LONGITUDE", "Opening month"]]

        else:
            amenity_coordinates = amenity_coordinates[["address","LATITUDE", "LONGITUDE"]]

        radius = self.params[amenity]["radius"]
        no_of_amenities_within_radius = f"no_of_{amenity}_within_{radius}_km"
        distance_to_nearest_amenity = f"distance_to_nearest_{amenity}"

        amenity_features = pd.DataFrame(hdb_data.progress_apply(lambda row: hdb_est.utils.find_nearest_amenity(row,
                                                                                                               amenity_coordinates,
                                                                                                               radius = radius,
                                                                                                               period = period),
                                                                                                               axis = 1).tolist(),
                                                                                                               columns=[no_of_amenities_within_radius,
                                                                                                                        distance_to_nearest_amenity])
        
        return amenity_features


    


