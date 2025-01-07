# Cryptocurrency Price Prediction Using LSTM

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Dataset and Preprocessing](#dataset-and-preprocessing)
4. [Model Architecture](#model-architecture)
5. [Workflow Overview](#workflow-overview)
6. [Results and Visualizations](#results-and-visualizations)
7. [Technologies Used](#technologies-used)
8. [Future Enhancements](#future-enhancements)

---

## Introduction
The cryptocurrency market is characterized by its high volatility and unpredictability. Accurate price forecasting can help investors navigate this challenging environment. This project employs LSTM, a type of recurrent neural network (RNN), designed to handle sequential data, to predict cryptocurrency price trends.

By training the model on historical price data, the project provides:
- Insightful visualizations of actual vs. predicted prices.
- A reliable method for forecasting future prices.
- Metrics to evaluate prediction accuracy.

---

## Features
- Data Preprocessing: Cleans and scales the data for accurate modeling.
- LSTM Model: A multi-layered architecture designed for sequential data analysis.
- Prediction Capabilities: Forecasts future cryptocurrency prices for the next 30 days.
- Visualization: Graphs to compare actual and predicted prices, and forecasted trends.
- Performance Metrics: Uses RMSE to evaluate the model's prediction accuracy.

---

## Dataset and Preprocessing
- Dataset: The project uses historical cryptocurrency price data with fields like `Open`, `High`, `Low`, `Close`, and `Volume`.
- Preprocessing Steps:
  - Extract the `Close` price for analysis.
  - Scale the data using MinMaxScaler to ensure values fall between 0 and 1.
  - Split the data into training (75%) and testing (25%) subsets.
  - Create sequences of 100 time steps as input for the LSTM model.

---

## Model Architecture
The LSTM model used in this project consists of:
- Three LSTM Layers: Captures long-term dependencies in the time-series data.
- Dense Layer: Outputs a single value representing the predicted price.
- Optimizer and Loss Function: Uses the Adam optimizer and Mean Squared Error (MSE) loss function.

Model Summary:
```python
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
```

---

## Workflow Overview
1. Data Loading: Load the cryptocurrency dataset and inspect its structure.
2. Preprocessing: 
   - Scale data and split it into training and testing sets.
   - Create sequences for LSTM input.
3. Model Training:
   - Train the LSTM model using the training data.
   - Evaluate the model on test data.
4. Prediction and Visualization:
   - Predict prices for test data and visualize actual vs. predicted values.
   - Forecast future prices (30 days) and visualize the trend.

---

## Results and Visualizations
### Key Results:
- Prediction Accuracy: The model achieved low RMSE values, indicating accurate predictions.
- Forecasting: The model provided a clear 30-day forecast of cryptocurrency prices.

### Visualizations:
1. Actual vs. Predicted Prices:  
   A graph comparing actual and predicted prices during the test phase.  
   ![Actual vs Predicted Prices](path-to-your-plot.png)

2. Future Price Forecasting:  
   A graph showing the next 30 days' predicted prices.  
   ![Future Price Forecast](path-to-your-forecast-plot.png)

---

## Technologies Used
- Programming Language: Python
- Libraries and Tools:
  - Pandas, NumPy: Data manipulation and analysis.
  - Matplotlib: Data visualization.
  - Scikit-learn: Data scaling.
  - TensorFlow/Keras: Model building and training.

---

## Future Enhancements
1. Sentiment Analysis: Include real-time sentiment data from social media to enhance predictions.
2. Blockchain Analytics: Integrate transaction data to uncover deeper market insights.
3. Model Comparison: Experiment with advanced architectures like GRU or Transformer models for better accuracy.
4. Broader Application: Extend the model to predict prices for multiple cryptocurrencies simultaneously.

