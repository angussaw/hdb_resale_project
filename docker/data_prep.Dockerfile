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

COPY --chown=$ID:$ID data/ data/
COPY --chown=$ID:$ID src/ src/
COPY --chown=$ID:$ID conf/ conf/
CMD python src/data_prep_pipeline.py