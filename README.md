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


# 1. Environment


### Please ensure dependencies adhere to python 3.10
```cmd
conda env create -f conda-env.yaml
conda activate xnn-env
```

```cmd
pip install requirements.txt
```

# 2. Web scraping for amenity coordinates

```
python scripts/amenities.py
```



# 3. Data Preparation

### Optional: Creating Postgres table for data preparation
```sql
CREATE TABLE <table_name_for_storing_derived_features> (
    month INTEGER,
    town VARCHAR(50),
    flat_type VARCHAR(50),
    block VARCHAR(50),
    street_name VARCHAR(100),
    storey_range VARCHAR(50),
    floor_area_sqm FLOAT,
    flat_model VARCHAR(50),
    lease_commence_date INTEGER,
    remaining_lease VARCHAR(50),
    resale_price FLOAT,
    cpi FLOAT,
    region VARCHAR(50),
    year_month VARCHAR(50),
    year INTEGER,
    lease_age INTEGER,
    latitude FLOAT,
    longitude FLOAT,
    no_of_malls_within_2_km INTEGER,
    distance_to_nearest_malls FLOAT,
    no_of_schools_within_2_km INTEGER,
    distance_to_nearest_schools FLOAT,
    no_of_parks_within_2_km INTEGER,
    distance_to_nearest_parks FLOAT,
    "no_of_MRT_stations_within_2_km" INTEGER,
    "distance_to_nearest_MRT_stations" FLOAT, 
    date_context VARCHAR(50)
);
```

### If saving to postgres, assuming the postgres database and table exists with the correct schema:

```yaml
### conf/data_prep.yaml
files:
    save_to_source: "postgres"
    derived_features_table_name: <insert postgres table name to save derived features to>
```

```cmd
set DATE=<current date: "yyy-mm-dd">
set POSTGRES_USER=<insert postgres user>
set POSTGRES_PWD=<insert postgres password>
set POSTGRES_PORT=<insert postgres port>
set POSTGRES_DB=<insert postgres database>
set POSTGRES_HOST=host.docker.internal

docker build -t hdb_data_prep:0.1.0 -f docker/data_prep.Dockerfile .  
```

```cmd
docker run --rm --name hdb_data_prep -e DATE=%DATE% -e POSTGRES_USER=$POSTGRES_USER -e POSTGRES_PWD=$POSTGRES_PWD -e POSTGRES_HOST=$POSTGRES_HOST -e POSTGRES_PORT=$POSTGRES_PORT -e POSTGRES_DB=$POSTGRES_DB --add-host=host.docker.internal:host-gateway hdb_data_prep:0.1.0
```


### Else if saving to a csv file to data/preprocessed/for_training:

```yaml
### conf/data_prep.yaml
files:
    save_to_source: "csv"
    preprocessed_save_path: "data/preprocessesd/for_training/hdb_preprocessed.csv"
```

```cmd
set DATE=<current date: "yyy-mm-dd">

docker build -t hdb_data_prep:0.1.0 -f docker/data_prep.Dockerfile .  
```

```cmd
docker run --rm --name hdb_data_prep -e DATE=%DATE% hdb_data_prep:0.1.0
```


# 4. Training

### Setting up mlflow server
```cmd
docker build -t mlflow-server -f docker/mlflow.Dockerfile .

docker run --name mlflow_server -p <insert port for mlflow server>:<insert port for mlflow server> -v $(pwd)/mlflow:/mlflow mlflow-server 
```

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

```cmd
set MLFLOW_TRACKING_URI=http://host.docker.internal:<insert port for mlflow server>

set POSTGRES_USER=<insert postgres user>
set POSTGRES_PWD=<insert postgres password>
set POSTGRES_PORT=<insert postgres port>
set POSTGRES_DB=<insert postgres database>
set POSTGRES_HOST=host.docker.internal

docker build -t hdb_training:0.1.0 -f docker/training.Dockerfile .  
```

```cmd
docker run --rm --name hdb_training -e POSTGRES_USER=$POSTGRES_USER -e POSTGRES_PWD=$POSTGRES_PWD -e POSTGRES_HOST=$POSTGRES_HOST -e POSTGRES_PORT=$POSTGRES_PORT -e POSTGRES_DB=$POSTGRES_DB -e MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI --add-host=host.docker.internal:host-gateway -v $(pwd)/mlflow:/mlflow hdb_training:0.1.0
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


```cmd
set MLFLOW_TRACKING_URI=http://host.docker.internal:<insert port for mlflow server>

docker build -t hdb_training:0.1.0 -f docker/training.Dockerfile .  
```

```cmd
docker run --rm --name hdb_training hdb_training:0.1.0 -e MLFLOW_TRACKING_URI=%MLFLOW_TRACKING_URI% --add-host=host.docker.internal:host-gateway -v $(pwd)/mlflow:/mlflow hdb_training:0.1.0
```

# 5. Deployment


```dockerfile
### docker/fast_api.Dockerfile

ENV MLFLOW_TRACKING_URI=http://host.docker.internal:<insert port for mlflow server>
ENV MODEL_URI=<insert mlfow model uri of chosen model>
ENV RUN_ID=<insert mlfow run_id of chosen model>
```

```cmd
docker-compose up -d --build
```



