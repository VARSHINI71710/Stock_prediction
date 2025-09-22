import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# -------------------------------
# Paths
# -------------------------------
MODEL_PATH = "stock_model.h5"
SCALER_PATH = "scaler_close.pkl"  # scaler for closing prices only

# -------------------------------
# Load Model
# -------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} not found. Place your trained model here.")

model = load_model(MODEL_PATH, compile=False)  # Avoid metrics deserialization error

# -------------------------------
# Load or create Scaler
# -------------------------------
if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
    print("✅ Loaded existing scaler.")
else:
    # If not found, create a dummy scaler (will need proper training data later)
    print("⚠ scaler_close.pkl not found! Creating a dummy scaler (fit later with training data).")
    scaler = MinMaxScaler(feature_range=(0,1))
    dummy_data = np.arange(100).reshape(-1,1)  # just to avoid errors
    scaler.fit(dummy_data)
    joblib.dump(scaler, SCALER_PATH)
    print("✅ Dummy scaler saved. Replace with real scaler after training.")

# -------------------------------
# Prediction function
# -------------------------------
def predict_stock(sequence_text):
    """
    Input: comma-separated last 60 closing prices
    Output: predicted next closing price
    """
    try:
        # Convert text input to list of floats
        sequence = [float(x.strip()) for x in sequence_text.split(",")]
        
        if len(sequence) != 60:
            return "Error: Please enter exactly 60 values."
        
        # Scale sequence
        seq_scaled = scaler.transform(np.array(sequence).reshape(-1,1))
        seq_scaled = np.reshape(seq_scaled, (1, seq_scaled.shape[0], 1))
        
        # Predict
        pred_scaled = model.predict(seq_scaled)
        pred = scaler.inverse_transform(pred_scaled)
        
        return round(float(pred[0][0]), 2)
    
    except Exception as e:
        return f"Error: {str(e)}"

# -------------------------------
# Gradio Interface
# -------------------------------
interface = gr.Interface(
    fn=predict_stock,
    inputs=gr.Textbox(
        label="Enter last 60 closing prices (comma-separated)",
        placeholder="e.g., 150.12,150.45,150.78,..."
    ),
    outputs=gr.Textbox(label="Predicted Next Closing Price"),
    title="S&P 500 Stock Price Predictor",
    description="Enter the last 60 closing prices of a stock to predict the next closing price"
)

# -------------------------------
# Launch App
# -------------------------------
interface.launch()
