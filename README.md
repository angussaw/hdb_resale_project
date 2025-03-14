# Estimating and interpreting HDB resale prices via feature engineering and SHAP values

Download the dataset from [here](https://drive.google.com/drive/folders/1dRHd3EO5lujuH7JW8pcNTBXARE69vcs9?usp=sharing)

# Folders/File structure 
```bash
├── data
│   ├── eda
│   │    └── flat_coordinates.csv
│   ├── for_data_cleaning
│   │    └── cpi.csv
│   ├── for_feature_engineering
│   │    ├── malls
│   │    │     └── mall_coordinates.csv
│   │    ├── mrt_stations
│   │    │     └── mrt_station_coordinates_w_period.csv
│   │    ├── parks
│   │    │     └── park_coordinates.csv
│   │    └── schools
│   │          └── school_coordinates.csv
│   ├── preprocessed
│   │    └── for_training
│   └── raw
│        └── for_training
│              ├── resale-flat-prices-based-on-registration-date-from-jan-2015-to-dec-2016.csv
│              └── resale-flat-prices-based-on-registration-date-from-jan-2017-onwards.csv
```

```bash
├── conf
│   ├── logging.yaml
│   ├── data_prep.yaml
│   └── train_config.yaml
```

```bash
├── docker
│   ├── data_prep.Dockerfile
│   ├── training.Dockerfile
│   ├── fast_api.Dockerfile
│   └── streamlit_app.Dockerfile
```

```bash
├── notebooks
│   └── eda.ipynb
```

```bash
├── scripts
│   └── amenities.py
```

```bash
├── src
│   ├── data_prep_pipeline.py
│   ├── train_pipeline.py
│   ├── fast_api.py
│   ├── streamlit_app.py
│   ├── streamlit_app_local.py
│   ├── streamlit_app_demo.py
│   └── hdb_resale_estimator
│          ├── __init__.py
│          ├── utils.py
│          ├── data_prep
│          │     ├── __init__.py
│          │     ├── data_cleaning.py
│          │     ├── feature_engineering.py
│          │     └── data_preparation.py
│          └── modeling
│               ├── __init__.py
│               ├── builder.py
│               ├── model.py
│               ├── evaluation.py
│               ├── train_test_split.py
│               └── training.py
```


# 1. Setup

## a. Conda environment

### Please ensure dependencies adhere to python 3.10
```bash
conda create -n hdb-app python=3.10
conda env update -n hdb-app -f conda-env.yaml
conda activate hdb-app
```

```bash
pip install requirements.txt
```

## b. Environment Variables

### Fill in `.env` file with the relevant values for the following environment variables:

```
POSTGRES_USER=<insert postgres user>
POSTGRES_PWD=<insert postgres password>
POSTGRES_PORT=<insert postgres port>
POSTGRES_DB=<insert postgres database>
POSTGRES_HOST=host.docker.internal

MLFLOW_TRACKING_URI=http://host.docker.internal:5005
EXPERIMENT_ID=<insert mlflow experiment id for inference deployment>
RUN_ID=<insert mlflow run id for inference deployment>
```

## c. Setting up PostgreSQL and MLflow services

### Make sure your `.env` file contains the necessary environment variables, then run:

```bash
docker-compose -f docker-compose-setup.yaml up -d
```

### After the services are up, run the necessary alembic migrations to create the tables required

```bash
alembic upgrade head
```

## d. Optional: Web scraping for amenity coordinates

```bash
python scripts/amenities.py
```


# 3. Data Preparation

### If saving to postgres, assuming the postgres database and table exists with the correct schema:

```yaml
### conf/data_prep.yaml
files:
    save_to_source: "postgres"
    derived_features_table_name: <insert postgres table name to save derived features to>
```

```bash
docker build -t hdb_data_prep:0.1.0 -f docker/data_prep.Dockerfile .  
```

```bash
source .env

docker run --rm --name hdb_data_prep --env-file .env --add-host=host.docker.internal:host-gateway hdb_data_prep:0.1.0
```


### Else if saving to a csv file to data/preprocessed/for_training:

```yaml
### conf/data_prep.yaml
files:
    save_to_source: "csv"
    preprocessed_save_path: "data/preprocessesd/for_training/hdb_preprocessed.csv"
```

```bash
docker build -t hdb_data_prep:0.1.0 -f docker/data_prep.Dockerfile .  
```

```bash
source .env

docker run --rm --name hdb_data_prep --env-file .env hdb_data_prep:0.1.0
```


# 4. Training

### If reading derived features from postgres table, assuming the postgres database and table exists with the correct schema::

```yaml
### conf/train_config.yaml
files:
  derived_features:
    read_from_source: postgres
    postgres_params:
      table_name: <insert postgres table name to read derived features from>
      columns:
        - month
        - year
        - flat_type
        - storey_range
        - floor_area_sqm
        - flat_model
        - lease_age
        - no_of_malls_within_2_km
        - distance_to_nearest_malls
        - no_of_schools_within_2_km
        - distance_to_nearest_schools
        - no_of_parks_within_2_km
        - distance_to_nearest_parks
        - no_of_MRT_stations_within_2_km
        - distance_to_nearest_MRT_stations
        - region
        - resale_price
```

```bash
docker build -t hdb_training:0.1.0 -f docker/training.Dockerfile .  
```

```bash
source .env

docker run --rm --name hdb_training --env-file .env --add-host=host.docker.internal:host-gateway -v $(pwd)/mlflow:/mlflow hdb_training:0.1.0
```



### Else if reading from a csv file in data/preprocessed/for_training:

```yaml
### conf/train_config.yaml
files:
  derived_features:
    read_from_source: csv
    csv_params:
      data_path: "data/preprocessed/for_training/hdb_preprocessed.csv"
      concat: False
```


```bash
docker build -t hdb_training:0.1.0 -f docker/training.Dockerfile .  
```

```bash
source .env

docker run --rm --name hdb_training --env-file .env --add-host=host.docker.internal:host-gateway -v $(pwd)/mlflow:/mlflow hdb_training:0.1.0
```

# 5. Deployment

```bash
docker-compose -f docker-compose-inference.yaml up -d --build
```



