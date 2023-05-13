"""
## data_prep_pipeline retrieves the raw data and data prep configuration,
and initialises data preparation
"""
import logging

from hydra import compose, initialize

import hdb_resale_estimator as hdb_est

logger = logging.getLogger(__name__)

def main():
    with hdb_est.utils.timer("Data Preparation"):
        hdb_est.utils.setup_logging()
        with initialize(version_base=None, config_path="../conf"):
            data_prep_config = compose(config_name="data_prep")
            logger.info("Starting data preparation pipeline")
            logger.info("Retrieving raw data and data preparation config...")

            logger.info("Performing data preparation in training mode...")
            logger.info(
                "Reading raw data from %s...",
                data_prep_config["files"]["training"]["raw_data_path"],
            )
            raw_hdb_data = hdb_est.utils.read_data(
                data_path=data_prep_config["files"]["training"]["raw_data_path"], concat=True
            )
            logger.info("Shape of raw hdb data: %s", raw_hdb_data.shape)

            logger.info("Initialising data preparation...")
            hdb_preprocessed = hdb_est.data_prep.data_preparation.data_prep_pipeline(
                data_prep_config["data_prep"], raw_hdb_data
            )

            hdb_preprocessed.to_csv(data_prep_config["files"]["training"]["preprocessed_save_path"])

            number_of_nulls = hdb_preprocessed.isna().sum().sum()

        logger.info(f"Data preparation completed!!! There are {number_of_nulls} null values present")


if __name__ == "__main__":
    main()
