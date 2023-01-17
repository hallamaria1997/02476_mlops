FROM gcr.io/deeplearning-platform-release/pytorch-cpu

COPY requirements.txt requirements.txt

COPY setup.py setup.py

RUN pip install -r requirements.txt --no-cache-dir
