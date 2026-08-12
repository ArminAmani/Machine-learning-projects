# Stock Price Prediction with LSTM

A time-series forecasting project using a Long Short-Term Memory (LSTM) neural network to model historical stock-price patterns and generate future price predictions.

---

## 📌 Overview

This project explores the application of deep learning to financial time-series forecasting.

Historical stock-market data is processed and transformed into sequential samples that can be used by an LSTM network to learn temporal relationships between previous price movements and future closing prices.

The workflow includes:

- Historical stock-price data preparation
- Exploratory visualization
- Sequence-data augmentation
- Feature scaling
- Time-series train/test splitting
- Sliding-window sequence generation
- LSTM model development
- Model training and evaluation
- Predicted vs. actual price comparison
- Prediction from the latest available sequence

---

## 📊 Data

The analysis uses historical stock-market data containing common market variables such as:

- Open
- High
- Low
- Close
- Volume

The neural network uses the main price features:

    Open
    High
    Low
    Close

with the **closing price** used as the prediction target.

---

## ⚙️ Data Preparation

The time-series data is prepared through several preprocessing steps:

1. Date conversion and chronological organization
2. Visualization of historical price behavior
3. Sequence-data augmentation
4. Min-Max feature scaling
5. Chronological train/test splitting
6. Sliding-window sequence construction

A sequence length of **20 time steps** is used to construct the input samples for the recurrent neural network.

---

## 🧠 LSTM Model

The forecasting model is based on a stacked LSTM architecture:

    Input Sequence
          ↓
    LSTM Layer
          ↓
    LSTM Layer
          ↓
    Dropout
          ↓
    Dense Output

The model uses two LSTM layers to capture temporal dependencies in the stock-price sequence, followed by dropout for regularization and a dense output layer for price prediction.

The network is trained using:

- Adam optimizer
- Mean Squared Error loss
- Regression-based performance monitoring

---

## 📈 Evaluation

Model performance is examined through numerical evaluation and visual comparison between observed and predicted stock prices.

The notebook includes:

- Training and validation loss analysis
- Prediction on the held-out test sequence
- Comparison of predicted and observed prices
- Visualization of forecasting behavior
- Prediction using the most recent available sequence

The results provide a practical demonstration of how recurrent neural networks can be applied to sequential financial data and time-series forecasting problems.

---

## 💡 Key Concepts

This project demonstrates several important concepts in deep learning and time-series modeling:

- Sequential data preparation
- Temporal dependency modeling
- Sliding-window sampling
- Feature normalization
- Recurrent neural networks
- LSTM-based forecasting
- Dropout regularization
- Regression evaluation
- Time-series visualization

---

## 🧰 Technologies

- Python
- Jupyter Notebook
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Statsmodels

---

## 📁 File

- [`Stock-Price-Prediction.ipynb`](./Stock-Price-Prediction.ipynb) — Complete implementation of the data preprocessing, sequence generation, LSTM training, evaluation, visualization, and prediction workflow.