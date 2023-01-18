# Base image

FROM python:3.7-slim

# install python

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# copying essential parts into container

COPY requirements.txt requirements.txt

COPY setup.py setup.py

COPY data.dvc data.dvc

COPY .dvcignore .dvcignore

COPY src/ src/

COPY reports/ reports/

COPY models/ models/

# setting working directory

WORKDIR /

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install 'dvc[gs]'
RUN dvc init --no-scm
RUN dvc remote add -d myremote gs://dtumlops-twitter-sentiment-data/

RUN dvc pull -v
COPY data/ data/

COPY src/API/main.py src/API/main.py

CMD exec uvicorn main:app --port $PORT --host 0.0.0.0 --workers 1