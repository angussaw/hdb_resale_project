"""
feature_engineering.py will contain the neccessary FeatureEngineer class to perform feature engineering
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
    """
    FeatureEngineer class contains methods to calculate/extract
    new derived features from existing features of each hdb flat
    transaction.

    Utilizes configuration parameters to generate 
    these new features
    """

    def __init__(self, params: dict)-> None:
        self.month_feature = params["month"]
        self.feature_engineering_params = params["feature_engineering"]
        self.year_feature = self.feature_engineering_params["year"]
        self.year_month_feature = self.feature_engineering_params["year_month"]

    def engineer_features(self, hdb_data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features from cleaned hdb data

        Returns:
           pd.DataFrame: Output dataframe containing
           each hdb transaction and their respective derived features
        """
        hdb_data = hdb_data.head(50)
        
        logger.info("Mapping towns to regions...")
        hdb_data = self.map_regions(hdb_data, self.feature_engineering_params["map_regions"])

        logger.info("Extracting transaction's year and month...")
        hdb_data = self.extract_year_month(hdb_data, self.feature_engineering_params["extract_year_month"])

        logger.info("Calculating lease age...")
        hdb_data = self.calculate_lease_age(hdb_data, self.feature_engineering_params["calculate_lease_age"])

        logger.info("Generating amenity features...")
        amenity_features_list = self.generate_amenities_features(hdb_data, self.feature_engineering_params["generate_amenities_features"])

        logger.info("Merging hdb derived features...")
        derived_features_hdb = pd.concat([hdb_data,
                                          amenity_features_list], axis = 1)

        return derived_features_hdb
    

    def map_regions(self, hdb_data: pd.DataFrame, params: dict) -> pd.DataFrame:
        """Function to map each town to its respective region

        Args:
            hdb_data (pd.DataFrame): Dataframe containing each hdb transaction
            params (dict): Config params

        Returns:
            pd.DataFrame: Dataframe containing each hdb transaction with regions feature
        """
        region_feature = params['region']
        town_feature  = params['town']
        map_regions = params['map_regions']

        hdb_data[region_feature] = hdb_data[town_feature].map({town: region for region, towns in map_regions for town in towns})

        return hdb_data
    
    def extract_year_month(self, hdb_data: pd.DataFrame) -> pd.DataFrame:
        """Function to extract the transaction's respective year and month

        Args:
            hdb_data (pd.DataFrame): Dataframe containing each hdb transaction

        Returns:
            pd.DataFrame: Dataframe containing each hdb transaction with year and month features
        """

        hdb_data[self.year_month_feature] = hdb_data[self.month_feature]
        hdb_data[self.month_feature] = hdb_data[self.year_month_feature].dt.month
        hdb_data[self.year_feature] = hdb_data[self.year_month_feature].dt.year

        return hdb_data
    
    def calculate_lease_age(self, hdb_data: pd.DataFrame, params: dict) -> pd.DataFrame:
        """Function to calculate the lease age of the hdb flat

        Args:
            hdb_data (pd.DataFrame): Dataframe containing each hdb transaction
            params (dict): Config params

        Returns:
            pd.DataFrame: Dataframe containing each hdb transaction with lease age feature
        """
        lease_age_feature = params['lease_age']
        lease_commence_date_feature = params['lease_commence_date']

        hdb_data[lease_age_feature] = hdb_data[self.year_feature] - hdb_data[lease_commence_date_feature]

        return hdb_data

    def generate_amenities_features(self, hdb_data: pd.DataFrame, params: dict) -> list:
        """Function to generate amenity features

        Args:
            hdb_data (pd.DataFrame): Dataframe containing each hdb transaction
            params (dict): Config params

        Returns:
            list: list of dataframes containing all amenity features
        """

        coordinates_feature = params['coordinates']
        block_feature = params['block']
        street_name_feature = params['street_name']
        amenities = params['amenities']

        logger.info("Generating lat long coordinates...")
        hdb_data[coordinates_feature] = hdb_data.progress_apply(lambda x: hdb_est.utils.find_coordinates(x[block_feature] + " " + x[street_name_feature]), axis = 1)
        amenity_features_list = []
        for amenity in amenities:
           logger.info(f"Getting nearest {amenity}...")
           feature_df = self.get_nearests_amenities(self, hdb_data, amenity, coordinates_feature, **params[amenity])
           amenity_features_list.append(feature_df)

        return amenity_features_list

    def get_nearests_amenities(self, hdb_data: pd.DataFrame, amenity: str, coordinates_feature, amenities_file_path: str, radius: int, period: bool) -> pd.DataFrame:
        """Function to get the following features for each flat:
                - no_of_amenities_within_radius
                - distance_to_nearest_amenity

        Args:
            hdb_data (pd.DataFrame): Dataframe containing each hdb transaction
            amenity (str): type of amenity (eg parks, schools, malls)
            coordinates_feature(str): name of coordinates feature
            amenities_file_path (str): file path containing the coordinates of each amenity location
            radius (int): radius around the flat
            period (bool): whether to take into account the opening date of the amenity

        Returns:
            pd.DataFrame: Dataframe containing the amenity-specific features 
        """
        amenity_details = hdb_est.utils.read_data(data_path = amenities_file_path, concat=False)
        if period:
            amenity_details = amenity_details.rename(columns={"Opening year": "YEAR", "Opening month": "MONTH"})
            amenity_details[self.year_month_feature] = pd.to_datetime(amenity_details[['YEAR', 'MONTH']].assign(DAY=1))
            amenity_details = amenity_details[["Name","LATITUDE", "LONGITUDE", self.year_month_feature]]

        else:
            amenity_details = amenity_details[["address","LATITUDE", "LONGITUDE"]]

        no_of_amenities_within_radius = f"no_of_{amenity}_within_{radius}_km"
        distance_to_nearest_amenity = f"distance_to_nearest_{amenity}"

        amenity_features = pd.DataFrame(hdb_data.progress_apply(lambda flat_transaction: hdb_est.utils.find_nearest_amenities(flat_transaction,
                                                                                                                              amenity_details = amenity_details,
                                                                                                                              radius = radius,
                                                                                                                              period = period,
                                                                                                                              coordinates_feature = coordinates_feature,
                                                                                                                              year_month_feature = self.year_month_feature),
                                                                                                                              axis = 1).tolist(),
                                                                                                                              columns=[no_of_amenities_within_radius,
                                                                                                                                       distance_to_nearest_amenity])
        
        return amenity_features


    

