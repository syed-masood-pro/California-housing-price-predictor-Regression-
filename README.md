# California Housing Price Predictor (Regression)

## Overview

This project builds a regression model using TensorFlow and Keras to predict the median house value for California districts based on 8 input features.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data Loading and Exploration](#data-loading-and-exploration)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Building](#model-building)
5. [Training the Model](#training-the-model)
6. [Visualizing Training Progress](#visualizing-training-progress)
7. [Evaluating the Model](#evaluating-the-model)
8. [Making Predictions](#making-predictions)
9. [Summary](#summary)
10. [Requirements](#requirements)
11. [Usage](#usage)
12. [Contributing](#contributing)
13. [License](#license)

## Project Overview

This project involves building a neural network to predict the median house value for California districts based on 8 input features:
- **MedInc:** Median income in block group
- **HouseAge:** Median house age in block group
- **AveRooms:** Average number of rooms per household
- **AveBedrms:** Average number of bedrooms per household
- **Population:** Block group population
- **AveOccup:** Average number of household members
- **Latitude:** Block group latitude
- **Longitude:** Block group longitude

## Data Loading and Exploration

We begin by loading the California housing dataset using `fetch_california_housing` from Scikit-Learn. The dataset is split into a full training set and a test set, and further split the full training set into a smaller training set and a validation set.

## Data Preprocessing

To improve model performance, we standardize the feature values using `StandardScaler`.

## Model Building

We build a Sequential neural network model using Keras with:
- One hidden layer (30 neurons with ReLU activation)
- An output layer for regression (predicting a continuous value)

## Training the Model

The model is compiled with Mean Squared Error loss function and SGD optimizer. It is trained for 20 epochs using the training set and validated on the validation set.

## Visualizing Training Progress

We plot the training and validation loss over epochs to monitor performance.

## Evaluating the Model

The model's performance is evaluated on the test set, achieving a test Mean Squared Error (MSE) of approximately 0.4333.

## Making Predictions

We select a few samples from the test set, make predictions, and compare them with the actual target values.

## Summary

In this project, we:
- Loaded and explored the California housing dataset
- Preprocessed the data by standardizing feature values
- Built a simple neural network model using TensorFlow and Keras
- Trained the model and monitored training progress
- Evaluated the model on test data
- Made predictions on new samples and compared with actual values

This comprehensive approach demonstrates the process of developing a regression model using TensorFlow and Keras, from data preparation through to model evaluation and prediction.

## Requirements

- TensorFlow
- Keras
- NumPy
- Matplotlib
- Pandas
- Scikit-Learn

## Usage

1. Clone the repository.
2. Install the required libraries.
3. Run the Jupyter notebook to build and train the model.
4. Evaluate the model and make predictions on new samples.

```bash

git clone https://github.com/syed-masood-pro/California-housing-price-predictor-Regression-.git
cd California-housing-price-predictor-Regression-
pip install -r requirements.txt
jupyter notebook house_price_prediction.ipynb
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss improvements, features, or bugs.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.


Feel free to customize this README file as needed. If you have any additional sections or modifications, just let me know!
