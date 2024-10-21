# Sentiment Analysis on IMDB Movie Reviews

## Overview

This project implements a **Sentiment Analysis** model to classify movie reviews from the **IMDB large movie review dataset** as either **positive** or **negative**. The dataset includes many labeled reviews, allowing the network to learn and generalize for sentiment classification.

The project involves:
- **Data Preprocessing** to prepare the raw text for use in NLP models.
- **Network Design** to train and test a model that classifies reviews based on their sentiment.
- **Model Saving** to allow for future testing and predictions.

## Dataset

The dataset used is the **IMDB Large Movie Review Dataset**, available for download [here](http://ai.stanford.edu/~amaas/data/sentiment/). It consists of movie reviews labeled as positive or negative.

## Preprocessing Steps

1. **Text Cleaning**: We clean the raw text data by:
   - Removing HTML tags, special characters, and unnecessary whitespace.
   - Lowercasing all text for uniformity.
   - Tokenizing the reviews into individual words.

2. **Stopwords Removal**: Common English words (stopwords) that don't contribute to the sentiment (like "and", "the", "is") are removed to reduce noise in the data.

3. **Tokenization and Padding**: 
   - The cleaned reviews are converted into sequences of integers, where each unique word is represented by an index.
   - The sequences are padded to ensure consistent input length for the model.

4. **Train-Test Split**: We use the provided split in the dataset, with separate folders for training and testing data.

## Network Design

The model is a custom **Neural Network** designed to classify the reviews as positive or negative. The architecture includes:

- **Embedding Layer**: Maps each word to a high-dimensional space.
- **Recurrent Layer (LSTM/GRU)**: Captures temporal dependencies in the sequence of words.
- **Dense Layers**: Final fully connected layers for sentiment classification.
- **Output Layer**: Uses a sigmoid activation function for binary classification.

### Training

- The model is trained using **Binary Crossentropy Loss**.
- **Adam Optimizer** is used for faster convergence.
- **Early Stopping** is employed to prevent overfitting.

### Testing

- The model is tested using the unseen data in the `test` folder.
- **Accuracy** is used as the performance metric.

## Results

- **Training Accuracy**: 94%
- **Testing Accuracy**: 90%

## Conclusion

This project demonstrates the process of building a custom sentiment analysis model from scratch. From preprocessing text data to training and testing a neural network, the model classifies movie reviews with a reasonable accuracy. The project provides insight into the importance of good preprocessing and model design when working with NLP tasks.

