02476 ML Ops -- Final Project
==============================

# Sentiment Classification of Tweets -- Project Description
- Halla María Hjartardóttir: s212958
- Hildur Margrét Gunnarsdóttir: s212951
- Rebekka Jóhannsdóttir: s212963
- Þórður Örn Stefánsson: s212957

## Overall Goal of The Project
The overall goal of this project is to combine the learning objectives of this ML Ops into a final product. After carrying out this project we want to know how to control the production machine learning life cycle with the assistance of the tools presented in this course. For the actual machine learning aspect of this project, the goal is to use NLP to determine whether the sentiment of a tweet is negative, positive, or neutral. 

## What framework are you going to use (PyTorch Image Models, Transformer, Pytorch-Geometrics)
Out of the three frameworks offered to work with, we will be using **Transformers** by the Huggingface group. This compliments our choice of project, NLP, since Transformers holds quantities of models intended for NLP tasks. 

## How do you intend to include the framework in your project
We vision taking advantage of the existing pre-trained models to compile a model that can successfully classify the sentiment of tweets into three classes using our data. At some point,we will look into how we can advance the pre-trained models to improve the results. 

After initial research on sentiment analysis using Transformers a good starting point is checking out pre-trained *DistilBERT, BERT, and RoBERTa* models for sentiment analysis. To classify based on the model result on our data we will use Transformers AutoModelForSequenceClassification class which adds a classification head on top of the pre-model outputs.

## What data are you going to run on (initially, may change)
The data used for this project is from the [Tweet Sentiment Extraction Kaggle challenge](https://www.kaggle.com/competitions/tweet-sentiment-extraction/data). The train split holds 27482 rows, while the test split holds 3535. Each data sample holds the following information: 

- **TextId**: unique identifier
- **Text**: a tweet.
- **Selected Text**: keywords of the tweet.
- Additionally the training data holds the column **Sentiment** which identifies the label of the tweet(positive, negative, or neutral)

This dataset is convenient for an NLP project since it holds a
 large amount of labeled data allowing us to verify the results of our model. The structure is also simple, which is optimal for a project that has to be implemented in such a brief timeline.

## What deep learning models do you expect to use
We expect that some experimenting is required to get started. Due to time constraints in this course, we have to utilize our time(and Transformers :) ) and use pre-trained models. Given our project goal, we will be using a pre-trained model called [twitter-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment). This is a RoBERTa-base model trained on ~124M tweets from January 2018 to December 2021 and finetuned for sentiment analysis with the TweetEval benchmark, ideal for the task of sentiment classification. 



Final Project in MLOps

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

