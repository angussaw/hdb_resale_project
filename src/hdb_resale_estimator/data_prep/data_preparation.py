"""
## data_preparation should contains the steps required to perform the following tasks:
## 1. Data Cleaning
## 2. Feature Engineering
"""
from datetime import datetime
import logging
import os

from omegaconf import DictConfig
import pandas as pd

import hdb_resale_estimator as hdb_est

logger = logging.getLogger(__name__)


def data_prep_pipeline(
    config: DictConfig,
    raw_hdb_data: pd.DataFrame,
    data_prep_mode: str = "training",
) -> None:
    """This is a wrapper function to clean the data and perform feature engineering to
    prep the data for model training and or inference"""

    if data_prep_mode == "training":
        # Clean data
        logger.info("Initialising Data cleaner...")
        data_cleaner = hdb_est.data_prep.data_cleaning.DataCleaner(
            raw_hdb_data=raw_hdb_data,
            params=config["data_prep"]["data_cleaning"],
        )

        logger.info("Cleaning data.....Please wait as this takes some time!!!")
        with hdb_est.utils.timer("Data Cleaning"):
            clean_hdb_data = data_cleaner.clean_data()
        logger.info("Data cleaning complete!")


        logger.info("Shape of clean hdb data: %s", clean_hdb_data.shape)

        return clean_hdb_data


