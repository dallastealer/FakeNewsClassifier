# FakeNewsClassifier

The goal of this project was to use NLP techniques to classify news articles as fake or real.

The dataset was found on kaggle (https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news) and the zipped version of the csv can be found in this repository.

In the jupyter notebook, I have conducted a series of modelling experiments to evaluate models for performing text classification.

## Techniques Used
* Encoding training labels as 0 and 1 to be used with the classification task and the binary crossentropy loss function
* Scikit-learn's TF-IDF vectorizer and Naive Bayes to establish a baseline model
* TensorFlow's TextVectorization layer to convert text to a numerical form which can be used within the model
* TensorFlow's Embedding layer to create trainable parameters for each vectorized word
* TensorFlow Hub to test pretrained word embeddings (Universal Sentence Encoder)
* Conversion of training and test arrays to TensorFlow Datasets which allow for batching and prefetching
* Dense models
* Transfer Learning (feature extraction) models
* LSTM models

## Results
The best model that was tested turned out to be a simple dense model with custom text vectorization and embedding layers. This model outperformed both LSTM models and feature extraction models with Universal Sentence Encoder, and performed with an accuracy of nearly 95% on testing data after only 8 epochs of training.
