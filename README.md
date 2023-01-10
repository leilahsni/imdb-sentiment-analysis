# IMDB Dataset - Sentiment Analysis
### Introduction

This repository contains the scripts and datasets used to train a supervised model for the task of sentiment analysis. More specifically, the model is trained to classify movie reviews based on the binary categories *positive* / *negative*.

The dataset used is the IMDb dataset which can be found [here] (https://datasets.imdbws.com). The original dataset comes in a directory containing two other directories, *neg* and *pos*. These directories contain respectively the negative review and positif review text files, which are aggregated in a single TSV file during the preprocessing phase.

The model gets a 94% accuracy with [*scikit-learn*](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)'s Multinomial Naive Bayes classifier.

### Run the script

In your virtual environement:

1.

> pip install -r requirements.txt

2.

> python imdb-classifier.py

The outfiles are :
	\- a model.pickle file
	\- a classification-model.txt file
	\- a confusion-matrix.png file
	\- a dataset.tsv file containing the entire annotated dataset