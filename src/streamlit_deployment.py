
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
import folium
from streamlit_folium import folium_static
from geopy.distance import geodesic
import yaml
from datetime import datetime
# import sys
# sys.path.append("conf")

# with open('conf/data_prep.yaml', 'r') as file:
#     config = yaml.safe_load(file)

logger = logging.getLogger(__name__)

def validate_input_data(input: dict):
    """_summary_

    Args:
        input (dict): _description_
    """

    messages = []

    if datetime.strptime(input["month"], '%Y-%m').date().year < input["lease_commence_date"]:
        messages.append("Lease commence year cannot be older than transaction year")

    if hdb_est.utils.find_coordinates(input["block"] + " " + input["street_name"]) == (float("inf"), float("inf")):
        messages.append("Please input a valid block and/or street name")

    return messages

# @hydra.main(config_path="../conf", config_name="data_prep.yaml")
def main():
    """
    This main function does the following:

    """

    logger = logging.getLogger(__name__)
    
    # logger.info("Intialise MLFlow...")
    # mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    # logger.info("Loading the model...")
    # builder = retrieve_builder(run_id=os.getenv("RUN_ID"),
    #                             model_uri=os.getenv("MODEL_URI"),
    #                             model_name=os.getenv("MODEL_NAME"))
    
    logger.info("Loading dashboard...")
    title = st.title('HDB Resale Price Estimator')

    with st.sidebar:
        st.write("Input flat details...")
        year = st.selectbox(
        'Select year',
        (["2015","2016","2017","2018","2019","2020"]))

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

        floor_area_sqm = st.number_input('Floor area in square meters', step=0.01)

        lease_commence_year = st.number_input('Lease commence year', step=1, value=1966)

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

    if st.button("Estimate resale price"):

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
                "remaining_lease":""}

        messages = validate_input_data(input)

        if len(messages) > 0:
            for message in messages:
                st.write(message)
        
        else:
            # input_data = pd.DataFrame([input])

            # logger.info("Conducting Data Prep...")

            # data_cleaner = hdb_est.data_prep.data_cleaning.DataCleaner(raw_hdb_data=input_data,
            #                                                             params=config["data_prep"],
            #                                                             inference_mode=True)
            # clean_input_data = data_cleaner.clean_data()

            # logger.info("Conducting Feature Engineering...")
            # feature_engineer = hdb_est.data_prep.feature_engineering.FeatureEngineer(
            #     params=config["data_prep"], inference_mode=True
            # )

            # derived_input_data = feature_engineer.engineer_features(
            #     hdb_data=clean_input_data)
            
            # predicted_resale_value = requests.post(url = "http://127.0.0.1:8500/predict",  data = json.dumps(derived_input_data.iloc[0].to_dict(), default=str))
            derived_input_data = requests.post(url = "http://127.0.0.1:8500/dataprep",  data = json.dumps(input, default=str))
            predicted_resale_value = requests.post(url = "http://127.0.0.1:8500/predict",  data = json.dumps(derived_input_data.json(), default=str))
            
            st.write("Predicted resale value: {}."
            .format(round(float(predicted_resale_value.text),0)))

            # map_display_coordinates = derived_input_data[["latitude","longitude"]]

            # nearest_amenity_coordinates_col = derived_input_data.columns[derived_input_data.columns.str.endswith("coordinates")]
            # for col in nearest_amenity_coordinates_col:
            #     nearest_amenity_coordinates_df = pd.DataFrame(derived_input_data[col].to_list(), columns=['latitude', 'longitude'])
            #     map_display_coordinates = pd.concat([map_display_coordinates,nearest_amenity_coordinates_df])

            # st.map(map_display_coordinates)

            derived_input_data = pd.DataFrame([derived_input_data.json()])

            flat_coordinates = [derived_input_data.iloc[0]["latitude"], derived_input_data.iloc[0]["longitude"]]
            flat_address = derived_input_data.iloc[0]["block"] + " " + derived_input_data.iloc[0]["street_name"]
            map = folium.Map(location=flat_coordinates, zoom_start=15)
            folium.Marker(
            location=flat_coordinates,
            popup=f"<b>{flat_address}/b>",
            tooltip = flat_address,
            icon=folium.Icon(color="blue", icon="home"),
            ).add_to(map)

            nearest_amenity_coordinates_col = derived_input_data.columns[derived_input_data.columns.str.endswith("coordinates")]
            icon_settings = {"MRT_stations": {"icon": "train", "color": "red"},
                            "schools": {"icon": "user-graduate", "color": "orange"},
                            "parks": {"icon": "tree", "color": "green"},
                            "malls":{"icon": "store", "color": "purple"}}
            for col in nearest_amenity_coordinates_col:

                amenity = col.replace("nearest_","").replace("_coordinates","")
                amenity_name = derived_input_data.iloc[0][f"nearest_{amenity}_name"]
                amenity_coordinates = derived_input_data.iloc[0][col]

                distance = round(float(str(geodesic(tuple(flat_coordinates),tuple(amenity_coordinates)))[:-3]),3)*1000
                displacement_coordinates = [tuple(flat_coordinates),
                                            tuple(amenity_coordinates)]
                folium.Marker(
                location=amenity_coordinates,
                popup=f"<b>{amenity_name}/<b>",
                tooltip = amenity_name,
                icon=folium.Icon(color=icon_settings[amenity]["color"], icon=icon_settings[amenity]["icon"], prefix="fa"),
                ).add_to(map)

                folium.PolyLine(displacement_coordinates, tooltip=f"{str(distance)} meters").add_to(map)

            folium_static(map, width=725)

    else:
        pass

if __name__ == "__main__":
    main()
    
    