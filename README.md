# Bank-Marketing-UCI-Dataset-Neural-Network

## Overview
This repository contains a Jupyter Notebook implementing a neural network for a bank subscription classification task. The project explores data preprocessing, model training, and evaluation using Python and deep learning libraries as part of the SWE303 Artificial Intelligence course.

## Dataset Link
https://archive.ics.uci.edu/dataset/222/bank+marketing

## Steps
- Data preprocessing
- Neural network model implementation
- Training and evaluation of the model
- Performance metrics visualization

## Methodology

A neural network deep learning model with backpropagation was employed to predict client subscription success, categorized as "yes" or "no." The dataset reflects a real-world scenario with a notable class imbalance, with 90% of responses labeled as "no" and 10% as "yes." To mitigate this issue, oversampling techniques were applied to balance the dataset. Additionally, hyperparameter tuning was performed to optimize the model's performance. This methodology aims to yield reliable predictions and offer valuable insights into client behaviors.

## Data Processing

Given the classification nature of the problem, a supervised model was utilized. The data preprocessing began with cleaning, addressing missing or invalid values, and replacing "999" entries with -1. Following the cleaning process, the dataset comprised 26,629 "no" responses and 3,859 "yes" responses. Categorical features were transformed using one-hot or label encoding, and the data was normalized with Standard Scaler to standardize the features. To address the class imbalance, oversampling was performed, and the data was shuffled to ensure a balanced dataset for model training.

## Neural Network (NN) Classification

The model architecture consists of five hidden layers with 128, 64, 32, 16, and 8 hidden units, which was determined through trial and error as an optimal configuration for hyperparameter tuning.

After preprocessing the data, the tests were divided into two main groups: one using the original unbalanced dataset and the other using a balanced dataset. For all tests, 300 epochs were maintained due to the complexity of the dataset and model. The data was split into 80% for training, 20% for testing, with a 10% validation split. Given that the dataset represents a binary classification problem (0 - 'no' or 1 - 'yes'), binary cross-entropy was chosen as the loss function, and the sigmoid activation function was applied to the output layer.

After experimenting with various parameter combinations for hyperparameter tuning, the highest accuracy was achieved with the following configuration: activation function: ReLU, epochs: 300, learning rate: 510⁻⁴, and batch size: 150. However, the accuracy and loss history graphs during training were not smooth, indicating model confusion and potential underlying issues. When using a lower learning rate of 510⁻⁵, although the accuracy slightly decreased, the training graphs became smoother.

