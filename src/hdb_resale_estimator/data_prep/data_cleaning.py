"""
data_cleaning.py will contain the neccessary DataCleaner class to clean, filter and impute the raw hdb data
"""
import logging
import pandas as pd

import hdb_resale_estimator as hdb_est

logger = logging.getLogger("__name__")

class DataCleaner:
    """DataCleaner class will be used to clean, impute and filter the hdb data"""

    def __init__(
        self, raw_hdb_data: pd.DataFrame, params: dict
    ) -> None:
        
        self.raw_hdb_data = raw_hdb_data
        self.month_feature = params["month"]
        self.data_cleaning_params = params["data_cleaning"]

    def clean_data(self) -> pd.DataFrame:
        """Takes in raw hdb data, performs cleaning and filtering
        of data.

        Returns:
            cleaned_df (pd.DataFrame): cleaned and filtered hdb data
        """
        
        self.remove_flat_types(self.data_cleaning_params["remove_flat_types"])
        self.replace_flat_models(self.data_cleaning_params["replace_flat_models"])
        self.change_dtype()
        self.adjust_resale(self.data_cleaning_params["adjust_resale"])

        return self.raw_hdb_data


    def remove_flat_types(self, params: dict):
        """Function to remove unwanted flat types

        Args:
            params (dict): Config params
        """
        flat_type_feature = params["flat_type"]

        self.raw_hdb_data = self.raw_hdb_data[~self.raw_hdb_data[flat_type_feature].isin(params["remove"])]

    def replace_flat_models(self, params: dict):
        """Function to replace flat models with specific values
        Args:
            params (dict): Config params
        """
        flat_model_feature = params["flat_model"]

        self.raw_hdb_data[flat_model_feature] = self.raw_hdb_data[flat_model_feature].apply(lambda x: x.upper())
        self.raw_hdb_data[flat_model_feature] = self.raw_hdb_data[flat_model_feature].replace(params["replace"])

    def change_dtype(self):

        """Function to change data types of certain columns
        """

        self.raw_hdb_data[self.month_feature] = pd.to_datetime(self.raw_hdb_data[self.month_feature])


    def adjust_resale(self, params: dict):
        """Function to factor in consumer price index into 
        hdb transaction resale price

        Args:
            params (dict): Config params
        """
        resale_price_feature = params["resale_price"]
        cpi_file_path = params["cpi_file_path"]

        cpi_data = hdb_est.utils.read_data(data_path = cpi_file_path, concat=False)
        cpi_data[self.month_feature] = pd.to_datetime(cpi_data[self.month_feature])
        self.raw_hdb_data = pd.merge(self.raw_hdb_data, cpi_data, how="left", on=self.month_feature)
        self.raw_hdb_data[resale_price_feature] = (self.raw_hdb_data[resale_price_feature] / self.raw_hdb_data['cpi']) * 100


        







    




