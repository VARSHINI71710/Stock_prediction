Stock prediction -- https://huggingface.co/spaces/Varshini-07/Stock_Close_prediction

📊 Stock Price Prediction using LSTM & Gradio
Overview

This project predicts the next-day closing price of a stock using a LSTM-based Recurrent Neural Network (RNN). It uses multiple features including open, high, low, close, volume and technical indicators such as moving averages. The prediction is presented via an interactive Gradio web app.

Features

Predict next-day closing price of any stock.

Uses multi-feature LSTM with 60-day sequences.

Includes technical indicators: MA5, MA10, MA20.

Early stopping implemented during training to prevent overfitting.

Interactive Gradio app to upload CSV files and get predictions.

Outputs predicted closing price based on latest stock data.

File Structure
/project-folder
│
├─ app.py             # Gradio web application
├─ stock_model.h5     # Trained LSTM model
├─ scaler.pkl         # MinMaxScaler used during training
├─ all_stocks_5yr.csv # Example dataset (historical stock prices)
└─ README.md          # Project documentation

app1.py file for the sample test code using large dataset.

Dataset

Dataset used: all_stocks_5yr.csv

Columns required:

date

Name (Stock symbol)

open

high

low

close

volume

Example: Yahoo Finance historical data or Kaggle’s “S&P 500 5-Year Stock Data”.

How It Works

Data Preprocessing:

Sorts data by date.

Selects the stock with the most occurrences (mode of Name).

Adds technical indicators: MA5, MA10, MA20.

Scales all features using MinMaxScaler.

Sequence Generation:

Uses the last 60 days of data to predict the next day's closing price.

Model:

LSTM layer with 100 units.

Dense layer outputting a single price value.

Compiled with Adam optimizer, MSE loss, and MAE metric.

Training:

80/20 Train/Test split.

EarlyStopping on validation loss (patience=5).

Prediction:

Gradio interface lets users upload a CSV.

Outputs predicted next closing price.


How to Run

python app.py

Open the URL shown in the terminal (http://127.0.0.1:7860).

Upload a CSV file containing stock data (columns: open, high, low, close, volume).

View the predicted next closing price.
Browser output:

📈 Predicted Next Close Price: 172.45


Shows the next day’s predicted closing price for the selected stock.

Optional: In future updates, a graph can be displayed showing last 60 days of actual prices + predicted price.

Metrics

Train/Test Accuracy (±1%)

Train/Test Direction Accuracy (UP/DOWN)

MAE & MSE used for model evaluation.

Future Improvements

Add visualization of actual vs predicted prices in Gradio.

Support prediction for multiple stocks at once.

Deploy as a web app on Hugging Face Spaces or Streamlit Cloud.

References

TensorFlow LSTM Documentation

Gradio Documentation

Kaggle Dataset: S&P 500 Stocks 5-Year Data
