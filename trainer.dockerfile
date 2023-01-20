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

COPY keys/ keys/

# setting environment variables

ENV WANDB_API_KEY=7d204c49a07b176284f3f17500c2f6081c919c75
ENV GOOGLE_APPLICATION_CREDENTIALS=/keys/dtumlops-tweet-sentiment-f74832ed9b30.json

# setting working directory

WORKDIR /

RUN pip install -r requirements.txt --no-cache-dir

RUN pip install dvc 'dvc[gs]'
RUN dvc init --no-scm
RUN dvc remote add -d myremote gs://dtumlops-twitter-sentiment-data/

RUN dvc pull -v 
COPY data/ data/

# naming as entrypoint

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]
CMD ["experiment=exp3"]
