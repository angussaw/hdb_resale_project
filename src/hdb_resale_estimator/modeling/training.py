"""Module containing the function to train a model
"""
import logging
import sys
import joblib
import mlflow
from omegaconf import DictConfig
import pandas as pd

import hdb_resale_estimator as hdb_est

logger = logging.getLogger(__name__)


def train_pipeline(
    config: DictConfig, feature_label_data: pd.DataFrame
) -> tuple[float, str]:
    """Main function to train a model. Instantiates a create a ClassicalModelBuilder object and sets model parameters specified in the Hydra config file
    
    Feature data is prepared by via several steps (encoding, splitting, scaling)

    Model is trained on training data and then evaluated on its performance
    on the test (and val) datasets

    Evaluation metrics and visualizations, and model artifacts are then logged to MLFlow

    Returns the model's performance on the specified optimization metric (eg recall)
    and the model uri

    Args:
        config (DictConfig): Configuration parameters for train pipeline
        feature_data (pd.DataFrame): Dataframe containing the derived features
        labels (pd.DataFrame): Dataframe containing the labels

    Returns:
        tuple[float, str]: Tuple containing the model's performance metric and the model uri
    """
    label_column = config["label_column"]
    if label_column in feature_label_data.columns:
        labels = feature_label_data[label_column]
        features = feature_label_data[[col for col in feature_label_data.columns if col != label_column]]
    else:
        raise(f"{label_column} not found in dataframe")


    chosen_model = config["model_params"]["chosen_model"]
    logger.info("Building %s model...", chosen_model)
    builder = hdb_est.modeling.builder.ClassicalModelBuilder().set_model(
        config["model_params"][chosen_model]["model_name"],
        config["model_params"][chosen_model]["params"],
    )

    logger.info("Processing training derived features...")
    if config["process_train_data"]["ordinal_encoding"]:
        logger.info("Performing ordinal encoding...")
        features = builder._ordinal_encode_variables(feature_data=features, columns=config["process_train_data"]["ordinal_encoding"])
        

    if config["model_params"][chosen_model]["one_hot_encode"]:
        logger.info("Performing one hot encoding on categorical features...")
        features = builder._one_hot_encode_cat_var(feature_data=features)

    # Split into train, test and validate data
    if config["process_train_data"]["train_test_val_split"]['test_size']:
        logger.info("Splitting Data into train, validate & test sets...")
        datasets = hdb_est.modeling.train_test_split.train_test_val_split(
            data=features,
            labels=labels,
            **config["process_train_data"]["train_test_val_split"],
        )
        logger.info(
        "Size of train set: %s, validate set: %s & test set: %s",
        len(datasets['train']['X']),
        len(datasets['val']['X']),
        len(datasets['test']['X']),
    )
    else:
        logger.info("Splitting Data into train & test sets...")
        datasets = hdb_est.modeling.train_test_split.train_test_val_split(
            data=features,
            labels=labels,
            **config["process_train_data"]["train_test_val_split"],
        )
        logger.info(
            "Size of train set: %s & test set: %s",
            len(datasets['train']['X']),
            len(datasets['test']['X']),
        )

    # Scaling data
    if config["model_params"][chosen_model]["scale_data"]:
        logger.info("Scaling the datasets...")
        datasets["train"]["X"] = builder.scale_data(datasets["train"]["X"].sort_index(axis=1), fitted_scaler = None)
        datasets["test"]["X"] = builder.scale_data(datasets["test"]["X"].sort_index(axis=1), fitted_scaler = builder.objects["standard_scaler"]["scaler"])
        try:
            datasets["val"]["X"] = builder.scale_data(datasets["val"]["X"].sort_index(axis=1), fitted_scaler = builder.objects["standard_scaler"]["scaler"])
        except:
            pass
    
    return pd.DataFrame(datasets["train"]["X"])
