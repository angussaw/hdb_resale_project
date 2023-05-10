"""Call clean_data function to clean, impute and hdb data.
"""
import logging
from datetime import datetime
import os
import pandas as pd
import numpy as np

import hdb_resale_estimator as hdb_est

logger = logging.getLogger("__name__")

class DataCleaner:
    """DataCleaner class will be used to clean, impute and filter the hdb data"""

    def __init__(
        self, raw_hdb_data: pd.DataFrame, params: dict
    ) -> None:
        """
        Args:
            raw_hdb_data (pd.DataFrame):
        """
        # Init raw data
        self.raw_hdb_data = raw_hdb_data
        self.params = params
        # self.inference_date = params["inference_date"]

        # self.phone_number = params["phone_number"]
        # self.trx_date = params["trx_date"]
        # self.itm_price = params["itm_price"]
        # self.itm_qty = params["itm_qty"]
        # self.trx_id = params["trx_id"]
        # self.trx_amt = params["trx_amt"]
        # self.disc_amt = params["disc_amt"]
        # self.bad_borrower = params["bad_borrower"]
        # self.open_date = params["open_date"]
        # self.dt_id = params["dt_id"]
        # self.kolektabilitas = params["kolektabilitas"]
        # self.months_before_loan_start = params["months_before_loan_start"]
        # self.data_cleaning_params = params["data_cleaning"]

    def clean_data(self) -> pd.DataFrame:
        """Takes in raw hdb data, performs cleaning and filtering
        of data.

        Returns:
            cleaned_df (pd.DataFrame): cleaned and filtered hdb data
        """
        self.filter_and_encode_flat_types()
        self.replace_flat_models()
        self.change_dtype()

        return self.raw_hdb_data


    def filter_and_encode_flat_types(self):
        """_summary_
        """
        self.raw_hdb_data = self.raw_hdb_data[~self.raw_hdb_data["flat_type"].isin(self.params["flat_type"]["flat_types_remove"])]
        self.raw_hdb_data["flat_type"] = self.raw_hdb_data["flat_type"].map(self.params["flat_type"]["flat_types_encode"])

    def replace_flat_models(self):
        """_summary_
        """
        self.raw_hdb_data["flat_model"] = self.raw_hdb_data["flat_model"].apply(lambda x: x.upper())
        self.raw_hdb_data["flat_model"] = self.raw_hdb_data["flat_model"].replace(self.params["flat_model"]["flat_model_replace"])

    def encode_storey_range(self):
        """_summary_
        """

        pass

    def change_dtype(self):
        """_summary_
        """
        self.raw_hdb_data["month"] = pd.to_datetime(self.raw_hdb_data["month"])


    




