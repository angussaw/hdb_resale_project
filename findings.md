# Estimating and interpreting HDB resale prices via feature engineering and SHAP explainability





## <b>Introduction</b>

The objective of this project is to develop an end-to-end machine learning pipeline to predict HDB resale prices based on existing features of each HDB flat transaction, as well as new derived features based on the geographical location of the flat. The end-to-end pipeline includes data preparation, feature engineering, model training and evaluation, and deployment.

For deployment, the model is deployed on a dashboard that users can input flat details to generate a predicted resale price. To understand the predicted resale prices made by the model during inference, SHAP values are calculated and plotted on the dashboard to determine which features contribute the most to the prediction. Geographical plots of the flats' vicinity are also generated to provide a visual explanation of how a flat's surrounding amenities contribute to its resale price.

## <b>Exploratory Data Analysis (EDA)</b>
### <u>Tools used</u>
- Pandas
- Matplotlib
- Seaborn
- Statistical tests
- API requests
- Webscraping tools (BeautifulSoup)

The HDB resale price data was downloaded from Data.gov.sg, containing approximately 120,000 resale transactions from 2015 to 2020. 

There are a total of 117,527 rows and 11 columns in the dataset. Each row represents a hdb resale transaction, and the target variable is the "resale_price" variable. 

The EDA selection consists of the following main tasks: 
1. Perform one-way ANOVA tests on raw categorical features with the target variable to check whether the unique categorical values have statistically significant differing mean resale prices (eg towns, flat model, flat type etc...)
   
2. Perform Pearson R tests on raw continous features with the target variable to check whether is it statistically significant that the distributions between the continous feature and the target feature are uncorrelated.
   
3. Identify features and the steps required clean and preprocess them in the data preparation pipeline (changing of data types, imputing null values, mapping to new categories, encoding etc...)
   
4. Perform webscraping to obtain list of amenities (malls, parks, mrt stations and schools) in Singapore, and make API calls to obtain the coordinates of each amenity, which will used to generate new features for each flat. Analyze each derived feature's relationship with the target resale price variable.

5. (Optional) Use phi-K correlation to obtain correlated feature pairs, which are those with correlation larger than defined correlation threshold, then obtain the columns with a lower correlation with label for removal

Here are the highlights of the EDA performed on the dataset. Please refer to the full notebook <i>"notebooks/eda.ipynb"</i> for more details about each feature




<u>Overview of raw features EDA</u>


| No | Feature             | Type            | Nulls present | Comments                                                                                                                                                                                                                  |
| -- | ------------------- | --------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1  | Month               | Object (string) | No            | \- In its raw form, need to convert it to datetime and extract out the year and month values                                                                                                                              |
| 3  | Town                | Object (string) | No            | \- significant relationship with target (one-way ANOVA)<br>\- High cardinality (26 unique values)<br>\- Will need to map to region to reduce cardinality                                                                  |
| 4  | Flat Type           | Object (string) | No            | \- very little samples with "MULTI-GENERATION" and "1 ROOM"<br>\- can consider dropping samples<br>\- majority flat type is "4 ROOM"<br>\- significant relationship with target (one-way ANOVA)                           |
| 5  | Block               | Object (string) | No            | \- too many unique values, likely not useful in predicting target variable                                                                                                                                                |
| 6  | Street Name         | Object (string) | No            | \- too many unique values, likely not useful in predicting target variable                                                                                                                                                |
| 7  | Flat Model          | Object (string) | No            | \- significant relationship with target (one-way ANOVA)<br>\- High cardinality (20 unique values)<br>\- Will need to consolidate common flat models together to reduce cardinality<br>\- majority flat model is "Model A" |
| 8  | Storey Range        | Object (string) | No            | \- significant relationship with target (one-way ANOVA)<br>\- May need to ordinal encode into numerical variable<br>\- majority storey range is "7 to 9"                                                                  |
| 9  | Floor Square Meter  | Float           | No            | \- strong positive correlation with target (pearson R)<br>\- mean floor area is 97.4sqm                                                                                                                                  |
| 10 | Lease Commence Date | Int             | No            | \- could use along with the year value it to calculate lease age, which has a negative correlation with target  (pearson R)                                                                                               |
| 11 | Remaining Lease     | Object (string) | No            | \- likely correlated with lease age and lease commence date<br>\- data quality is inconsistent (eg 86 years, 86, 86 years 07 months)                                                                                      |


<u>Overview of generated features EDA</u>

| No | Feature                         | Type  | Nulls present | Comments                                                                                                                                                                                                                    |
| -- | ------------------------------- | ----- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1  | no_of_schools_within_2_km       | Int   | No            | \- slight negative (-) correlation with the resale value of the flat<br>\- the greater the number of schools within 2km radius, the lower the resale value<br>\- mean number of schools within 2km radius is about 15           |
| 2  | distance_to_nearest_school      | Float | No            | \- slight positive (+) correlation with the resale value of the flat<br>\- the further away the flat is to its nearest school, the higher the resale value<br>\- mean distance to the nearest school is about 300m              |
| 3  | no_of_mrt_stations_within_2_km  | Int   | No            | \- slight positive (+) correlation with the resale value of the flat<br>\- the greater the number of mrt stations within 2km radius, the higher the resale value<br>\- mean number of mrt stations within 2km radius is about 3 |
| 4  | distance_to_nearest_mrt_station | Float | No            | \- slight negative (-) correlation with the resale value of the flat<br>\- the further away the flat is to its nearest mrt station, the lower the resale value<br>\- mean distance to the nearest mrt station is about 855m     |
| 5  | no_of_malls_within_2_km         | Int   | No            | \- slight positive (+) correlation with the resale value of the flat<br>\- the greater the number of malls within 2km radius, the higher the resale value<br>\- mean number of malls within 2km radius is about 5               |
| 6  | distance_to_nearest_mall        | Float | No            | \- slight negative (-) correlation with the resale value of the flat<br>\- the further away the flat is to its nearest mall, the lower the resale value<br>\- mean distance to the nearest mall is about 665m                   |
| 7  | no_of_parks_within_2_km         | Int   | No            | \- slight positive (+) correlation with the resale value of the flat<br>\- the greater the number of parks within 2km radius, the higher the resale value<br>\-  mean number of parks within 2km radius is almost 2             |
| 8  | distance_to_nearest_park        | Float | No            | \- slight negative (-) correlation with the resale value of the flat<br>\- the further away the flat is to its nearest park, the lower the resale value<br>\- mean distance to the nearest park is about 1.2km                  |

## <b>Data Preparation Pipeline</b>
### <u>Tools used</u>
- Pandas
- Hydra
- Geopy
- Docker
- Postgres SQL

The raw data is first read from the filepath and passed through the Data Cleaning module. The cleaned data is then passed through the Feature Engineering module, which generates derived features based on existing raw features as well as the coordinates of the various amenities. The derived features are then saved to a datatable in a Postgres database for storage.

The data preparation pipeline is containerized based on a docker image, please refer to the instructions on how to build the image and run the data preparation container.

![image info](/images/data_preparation_pipeline.png)

## <b>Training Pipeline</b>
### <u>Tools used</u>
- Scikit-learn
- MLFlow
- Optuna
- Hyperparameter tuning
- Docker

The derived features and target variable are first extracted from Postgres. The derived features then undergo a series of preprocessing steps that include ordinal encoding, one-hot encoding, train-validation-test splitting and standard scaling if necessary. 

The model is initialized with its specified hyperparameters and then trained on the training data. The trained model is then evaluated on the various datasets, generating visualizations and performance metrics. 

The model and its respective set of parameters, visualizations and metrics are then logged to MLFLow for tracking. Lastly, the metric to optimize the model on is returned at the end of the train pipeline.

The train pipeline is also containerized based on a docker image, please refer to the instructions on how to build the image and run the train container.

![image info](/images/train_pipeline.png)




## <b>Model Evaluation</b>

### <u>Performance metrics</u>
The following model architectures using ensemble learning (combining multiple weak learners into one predictive model) are trained:
- Random Forest
- Explainable Boosting Regressor
- XGBoost

For each model architecture, a baseline model is trained using all derived features and the default hyperparameters. The model's hyperparameters are then fine-tuned using Optuna to search for the optimal set of hyperparameters that minimizes the specified performance metric (validation room mean squared error). 

|                                | Baseline |            |         |            |       |            | Fine-tuned |            |         |            |       |            | Fine-tuned parameters                                                                                                                                                           |
| ------------------------------ | -------- | ---------- | ------- | ---------- | ----- | ---------- | ---------- | ---------- | ------- | ---------- | ----- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|                                | MAE      |            | RMSE    |            | r2    |            | MAE        |            | RMSE    |            | r2    |            |                                                                                                                                                                                 |
|                                | Train    | Validation | Train   | Validation | Train | Validation | Train      | Validation | Train   | Validation | Train | Validation |                                                                                                                                                                                 |
| Random Forest                  | 49963.5  | 49963.5    | 69247.1 | 68800.6    | 0.771 | 0.77       | 9711       | 18767.4    | 13282.7 | 26525.8    | 0.992 | 0.966      | \-max_depth: 20<br>\-min_samples_leaf: 1<br>\-n_estimators: 205                                                                                                                 |
| Explainable Boosting Regressor | 39630.6  | 39798.3    | 55222.2 | 55435.6    | 0.855 | 0.85       | 33511.2    | 34140      | 45915.1 | 46890.1    | 0.899 | 0.893      | \-inner_bags: 0<br>\-interactions: 10<br>\-learning_rate: 0.01<br>\-max_bins: 256<br>\- max_interaction_bins: 32<br>\-max_leaves: 3<br>\-min_samples_leaf: 2<br>\-outer_bags: 8 |
| XGBoost                        | 18948.8  | 21166      | 25637.9 | 29166.1    | 0.969 | 0.959      | 12427.8    | 17919.3    | 16657.6 | 25119.7    | 0.987 | 0.969      | \-learning_rate: 0.1<br>\-max_depth: 8<br>\-n_estimators: 480                                                                                                                   |

For the baseline models, XGBoost has the lowest validation root mean squared error of 29116, and the highest validation R squared score of 0.959. There is not much overfitting on the training set, as observed by the minor differences between the train and validation scores.

After fine tuning the hyperparameters to minimize the validation root mean squared error using the Optuna's TPE sampler, the XGBoost model's validation root mean squared error and R squared score decreased and increased to 25119 and 0.969 respectively. The fine-tuned XGBoost model was using a learning rate of 0.1 (default: 0.3), maximmum depth of 8 (default: 6) and 480 estimators (default: 100). 

The fine-tuned random forest model's validation root mean squared error and R squared score decreased and increased to 26525 and 0.966 respectively. It uses a max depth of 20, and 205 estimators (default: 100)

The selected models are also evaluated on the test set to ensure that the model is able estimate accurate predictions on new unseen data

|               | MAE     | RMSE    | r2    |
| ------------- | ------- | ------- | ----- |
| Random Forest | 18418.9 | 25936.6 | 0.968 |
| XGBoost       | 17816.8 | 24751.4 | 0.971 |






### <u>Feature importance</u>

SHAP values were calculated using the training set feature values, and were used to generate a summary plot to visualize which features have the greatest contribution to the final model prediction

![image info](/images/xgboost_plots.png)

Based on the summary plot for the XGBoost model, the top 5 features that have the highest SHAP values across all samples are "floor_area_sqm", "lease_age", "region_Central", "distance_to_nearest_MRT_stations" and "storey_range".

| Feature                          | Contribution to predicted value                                                                                                        |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| floor_area_sqm                   | The larger the floor area of a flat, the greater its contribution in increasing its predicted resale price, and vice versa             |
| lease_age                        | The older the flat, the greater its contribution in decreasing its predicted resale price, and vice versa                              |
| region_Central                   | If a flat is located in the central region, it increases the predicted resale price, and vice versa                                    |
| distance_to_nearest_MRT_stations | The closer the nearest MRT station is to a flat, the greater its contribution in increasing its predicted resale price, and vice versa |
| storey_range                     | The higher the flat is, the greater its contribution in increasing its predicted resale price, and vice versa                          |



## <b>Model deployment</b>
### <u>Tools used</u>
- Streamlit
- Folium
- FastAPI
- Docker

Between the various ensemble models that were trained, XGBoost model is selected for deployment as it has the best performance on the train and validation sets. It is also the fastest model to train out of the three models, which in turn takes the shortest time to generate the SHAP explainer for it. (With 20 features used, approximately 2^20 models are trained to generate the SHAP explainer)

The deployment is separated into two main components: the frontend where the user will interact with a dashboard by inputing flat details and view the visualizations generated, and the backend where the pipelines for data preparation, model prediction and SHAP values calculation will be performed. A Streamlit dashboard is built for the frontend, which will post requests to the backend via FastAPI to trigger the relevant pipelines whenever a user submits flat details to estimate its resale price.


![image info](/images/streamlit_fastapi.png)


1. <u>Validation of input data</u>: This ensures that the flat details submitted by the user is valid and suitable for model prediction (eg lease commence year cannot be older than transaction year)
2. <u>Post request to backend to perform data prep</u>: A request is sent to the backend to clean the input data and generate new derived features from it. It returns a dataframe consisting of derived features to the front end where it will be displayed (eg distances to nearest amenities, number of nearby amenities)
3. <u>Post request to backend to predict resale value</u>: Another request is sent to the back end to predict the resale value of a flat. If required, the data is encoded/scaled using the encoder/scaler object that was fitted on the training data. After which, the data is fed to the model to estimate the resale price. The model and the encoder/sacler objects are retrieved as artifacts from the respective MLFlow run
4. <u>Render map showing the hdb flat surroundings and nearby amenities</u>: Using the derived features, a geographical map is rendered on the dashboard displaying the names and locations of the nearest amenities that are within the flat's radius
5. <u>Post request to backend to generate shap values</u>: Another request is sent to the backend to calculate the SHAP values for the flat's derived features. It returns a SHAP explainer object back to the frontend
6. <u>Render waterfall plot of shap values to explain model prediction</u>: Using the SHAP explainer object, a waterfall plot is rendered on the dashboard to display the contributions that each feature value of the flat made to the final predicted resale price.



![image info](/images/dashboard_screenshot.png)


## <b>Future improvements</b>

1. As time goes by, the amenity details needs to be updated with the construction of new mrt stations and malls to ensure that the derived features are accurate.
2. For now, the opening date for the amenities is only accounted for MRT stations. Accounting the opening date for the other amenties would ensure that the derived features are more accurate.
3. New amenities that could possibly affect resale prices can be included in the feature engineering pipeline (eg medical centers, food centres, bus stops, interchanges etc)
4. Another round of feature selection can be performed by removing some features that have strong correlation with each other, as this might confound the model's explainability.
5. The target variable can be transformed by factoring in the yearly consumer price index before training to ensure thaat the model's predictions reflect the current economic conditions in Singapore
















