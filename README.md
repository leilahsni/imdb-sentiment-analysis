# IMDB Dataset - Sentiment Analysis
### Introduction

This repository contains the scripts and datasets used to train a supervised model for the task of sentiment analysis. More specifically, the model is trained to classify movie reviews based on the binary categories *positive* / *negative*.

The dataset used is the IMDb dataset which can be found [here] (https://datasets.imdbws.com). The original dataset comes in a directory containing two other directories, *neg* and *pos*. These directories contain respectively the negative review and positif review text files, which are aggregated in a single TSV file during the preprocessing phase.

The model gets a 94% accuracy with *scikit-learn*'s Multinomial Naive Bayes classifier.

### Run the script

In your virtual environement:

1.

> pip install -r requirements.txt

2.

> python classification_model.py

Two TSV files will be automatically created in data/tsv : deft-2009-test.tsv and deft-2009-train.tsv

On the first run, the model.sav file will be created automatically in the model directory. If a model.sav file already exists, you will be asked wether you want to overwrite the previous model or not : type *y* for yes, *n* for no.

The results (confusion matrix, classification report, model predictions, etc.) can all be found in the eval directory.