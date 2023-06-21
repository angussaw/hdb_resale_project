FROM python:3.10-slim-buster

ARG USER=user
ARG ID=1000
ARG REQUIREMENTS_TXT="requirements.txt"
ARG HOME_DIR="/home/$USER"

RUN groupadd -g $ID $USER && useradd -g $ID -m -u $ID -s /bin/bash $USER
WORKDIR $HOME_DIR
USER $USER

COPY --chown=$ID:$ID $REQUIREMENTS_TXT .
RUN pip3 install -r $REQUIREMENTS_TXT

COPY --chown=$ID:$ID src/ src/
COPY --chown=$ID:$ID conf/ conf/
COPY --chown=$ID:$ID models/ models/
COPY --chown=$ID:$ID data/for_feature_engineering data/for_feature_engineering

ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5005
ENV MODEL_URI=38d0ed447a714d01998318049ea8a905
ENV RUN_ID=934552010482666058

EXPOSE 8500

CMD python src/fast_api.py