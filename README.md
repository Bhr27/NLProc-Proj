# IMDB Perceptron Classifier

This project implements a simple Perceptron-based text classifier for sentiment analysis on the IMDB movie review dataset.

## Features

- Text preprocessing
- Stop word removal
- Train/test split
- Bag-of-words vectorization
- Perceptron classifier implementation
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score

## Dataset

IMDB Movie Review Dataset

## Notes

No machine learning libraries such as sklearn or PyTorch were used.

A subset of the dataset was used because the implementation relies on plain Python lists and becomes slow on large datasets.

## Run

```bash
python perceptron-classifier.py
