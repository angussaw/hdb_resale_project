FROM python:3.10-bullseye

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
COPY --chown=$ID:$ID images/ images/

EXPOSE 8501

ENTRYPOINT ["python", "-m", "streamlit", "run", "src/streamlit_app.py"]