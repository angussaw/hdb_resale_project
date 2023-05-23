from fastapi import FastAPI
import logging
import mlflow
import os
import joblib
import pandas as pd
import glob
import yaml

with open('conf/data_prep.yaml', 'r') as file:
    config = yaml.safe_load(file)

import sys
sys.path.append("src")
import hdb_resale_estimator as hdb_est

logger = logging.getLogger(__name__)

# Load model
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
model_uri = os.getenv("MODEL_URI")
run_id =  os.getenv("RUN_ID")
artifact_uri = f'mlflow-artifacts:/{run_id}/{model_uri}/artifacts/model'
logger.info("Downloading artifacts from MLFlow model URI: %s...", model_uri)
try:
    mlflow.artifacts.download_artifacts(
        artifact_uri=artifact_uri, dst_path="../models"
    )
except Exception as mlflow_error:
    logger.exception("Failed to load model: %s", mlflow_error)
    raise mlflow_error

logger.info("Artifact download successful")

model_path = glob.glob("../models/model/*.joblib")[0]
builder = joblib.load(model_path)

PRED_MODEL = builder.model
PRED_MODEL_FEATURES = builder.objects["features"] # before encoding


app = FastAPI()

@app.post("/predict")
def predict_resale_value(hdb_flat_dict: dict):
    """_summary_

    Args:
        hdb_flat_dict (dict): _description_
    """

    hdb_flat_df = pd.DataFrame([hdb_flat_dict])[PRED_MODEL_FEATURES]

    processed_hdb_flat_df = builder.process_inference_data(inference_data = hdb_flat_df)

    result = PRED_MODEL.predict(processed_hdb_flat_df)

    return result.tolist()[0]


@app.post("/dataprep")
def prepare_raw_data(input_data: dict):
    """_summary_

    Args:
        hdb_flat_dict (dict): _description_
    """

    input_data = pd.DataFrame([input_data])
    data_cleaner = hdb_est.data_prep.data_cleaning.DataCleaner(raw_hdb_data=input_data,
                                                                params=config["data_prep"],
                                                                inference_mode=True)
    clean_input_data = data_cleaner.clean_data()

    logger.info("Conducting Feature Engineering...")
    feature_engineer = hdb_est.data_prep.feature_engineering.FeatureEngineer(
        params=config["data_prep"], inference_mode=True
    )

    derived_input_data = feature_engineer.engineer_features(
        hdb_data=clean_input_data)
    
    return derived_input_data.iloc[0].to_dict()






