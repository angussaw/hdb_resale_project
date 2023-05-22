
import logging
import math
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import os
from pathlib import Path
import streamlit as st
import tempfile
import hdb_resale_estimator as hdb_est
import pandas as pd
import hydra
import requests
import json

logger = logging.getLogger(__name__)

@hydra.main(config_path="../conf", config_name="data_prep.yaml")
def main(config):
    """
    This main function does the following:
    - Loads trained model on cache
    - Gets image input from user to be loaded for inferencing
    - Conducts inferencing on image
    - Outputs prediction results on the dashboard
    - Generate heatmaps for each convolutional layer superimposed on image
    """

    logger = logging.getLogger(__name__)
    
    # logger.info("Intialise MLFlow...")
    # mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    # logger.info("Loading the model...")
    # builder = retrieve_builder(run_id=os.getenv("RUN_ID"),
    #                             model_uri=os.getenv("MODEL_URI"),
    #                             model_name=os.getenv("MODEL_NAME"))
    orig_cwd = hydra.utils.get_original_cwd()
    
    logger.info("Loading dashboard...")
    title = st.title('AIAP HDB Resale Price MLOps Exercise')

    year = st.selectbox(
    'Select year',
    (["2015","2016","2017","2018"]))

    month = st.selectbox(
    'Select month',
    (["01","02","03","04","05","06","07","08","09","10","11","12"]))

    month = year + "-" + month

    flat_type = st.radio(
    "Select flat type",
    ('3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE'))

    block = st.text_input("Block")

    street_name = st.text_input("Street Name")

    storey_range = st.selectbox(
    "Select storey range",
    ('01 TO 03','04 TO 06','07 TO 09','10 TO 12','13 TO 15','16 TO 18','19 TO 21',
    '22 TO 24','25 TO 27','28 TO 30','31 TO 33','34 TO 36','37 TO 39','40 TO 42',
    '43 TO 45','46 TO 48','49 TO 51'))

    floor_area_sqm = st.slider('Floor area in square meters', 31.0, 280.0)

    lease_commence_year = st.slider('Lease commence date', 1966, 2016)

    remaining_lease_years = st.text_input("Remaining lease years")
    remaining_lease_years_months = st.text_input("Remaining lease months")

    town_name = st.selectbox(
    "Select town",
    ('ANG MO KIO','BEDOK','BISHAN','BUKIT BATOK','BUKIT MERAH','BUKIT PANJANG','BUKIT TIMAH',
    'CENTRAL AREA','CHOA CHU KANG','CLEMENTI','GEYLANG','HOUGANG','JURONG EAST',
    'JURONG WEST','KALLANG/WHAMPOA','MARINE PARADE','PASIR RIS','PUNGGOL','QUEENSTOWN',
    'SEMBAWANG','SENGKANG','SERANGOON','TAMPINES','TOA PAYOH','WOODLANDS','YISHUN'))

    flat_model_name = st.selectbox(
    "Select flat model",
    ('2-room','Adjoined flat','Apartment','DBSS','Improved','Improved-Maisonette',
    'Maisonette','Model A','Model A-Maisonette','Model A2','Multi Generation',
    'New Generation','Premium Apartment','Premium Apartment Loft','Premium Apartment.',
    'Premium Maisonette','Simplified','Standard','Terrace','Type S1','Type S2'))

    if st.button("Predict resale price"):

        input = {"id":1,
                "month":month,
                "town":town_name,
                "flat_type": flat_type,
                "block":block.upper(),
                "street_name":street_name.upper(),
                "storey_range":storey_range,
                "floor_area_sqm":floor_area_sqm,
                "flat_model":flat_model_name,
                "lease_commence_date":lease_commence_year,
                "remaining_lease":f"{remaining_lease_years} years {remaining_lease_years_months} months"}

        input_data = pd.DataFrame([input])

        logger.info("Conducting Data Prep...")

        data_cleaner = hdb_est.data_prep.data_cleaning.DataCleaner(raw_hdb_data=input_data,
                                                                    params=config["data_prep"],
                                                                    inference_mode=True)
        clean_input_data = data_cleaner.clean_data()

        logger.info("Conducting Feature Engineering...")
        feature_engineer = hdb_est.data_prep.feature_engineering.FeatureEngineer(
            params=config["data_prep"], inference_mode=True, directory=orig_cwd
        )

        derived_input_data = feature_engineer.engineer_features(
            hdb_data=clean_input_data).iloc[0].to_dict()
        
        predicted_resale_value = requests.post(url = "http://127.0.0.1:8500/predict",  data = json.dumps(derived_input_data, default=str))
        
        st.write("Predicted resale value: {}."
        .format(predicted_resale_value.text))
    else:
        st.write("Awaiting a flat...")

if __name__ == "__main__":
    main()
    