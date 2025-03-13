FROM python:3.10-slim

# Install MLflow and dependencies
RUN pip3 install mlflow psycopg2-binary boto3

RUN mkdir /mlflow/

EXPOSE 5005

# Command to run MLflow server
CMD mlflow server \
    --host 0.0.0.0 \
    --port 5005 \
    --default-artifact-root file:/mlflow/artifacts \
    --backend-store-uri sqlite:///mlflow/mlflow.db