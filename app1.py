import gradio as gr
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError

model = load_model("stock_model1.h5", custom_objects={'mae': MeanAbsoluteError(), 'mse': MeanSquaredError()})
scaler = joblib.load("scaler1.pkl")


FEATURES = ['open', 'high', 'low', 'close', 'volume', 'MA5', 'MA10', 'MA20']

def predict_next_day(open_price, high_price, low_price, close_price, volume, ma5, ma10, ma20):
    """
    Predicts the next day's closing price using the trained LSTM model.
    """
    dummy_df = pd.DataFrame([[open_price, high_price, low_price, close_price, volume, ma5, ma10, ma20]],
                            columns=FEATURES)

    scaled_input = scaler.transform(dummy_df)

   
    reshaped_input = scaled_input.reshape(1, 1, len(FEATURES))

    scaled_prediction = model.predict(reshaped_input)[0][0]

    # To get the actual price, we need to inverse transform the prediction.
    # The scaler was trained on all features, so we need to create a dummy
    # array with zeros for all other features. The 'close' price is at index 3.
    # The inverse_close function from the original notebook is implemented here.
    def inverse_close(scaled_value, close_index=3, total_features=len(FEATURES)):
        dummy = np.zeros((1, total_features))
        dummy[:, close_index] = scaled_value
        return scaler.inverse_transform(dummy)[:, close_index]

    # Inverse transform the scaled prediction to get the real price
    predicted_price = inverse_close(scaled_prediction)[0]
    
    return f"${predicted_price:.2f}"

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_next_day,
    inputs=[
        gr.Number(label="Open Price", value=150.0),
        gr.Number(label="High Price", value=155.0),
        gr.Number(label="Low Price", value=149.0),
        gr.Number(label="Close Price", value=152.0),
        gr.Number(label="Volume", value=1000000),
        gr.Number(label="5-Day MA", value=151.5),
        gr.Number(label="10-Day MA", value=151.0),
        gr.Number(label="20-Day MA", value=150.5),
    ],
    outputs=gr.Textbox(label="Predicted Next Day's Close Price"),
    title="Stock Price Prediction",
    description="Enter today's stock data to predict tomorrow's closing price."
)

iface.launch(debug=True)
