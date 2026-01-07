import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Stock Price Prediction Dashboard", layout="wide")
st.title("Stock Price Prediction Dashboard")

start = '2015-01-01'
end = '2025-01-01'

st.sidebar.header("User Input Features")
def user_input_features():
    stock_symbol = st.sidebar.text_input("Stock Symbol", "KOTAKBANK.NS")
    return stock_symbol
stock_symbol = user_input_features()

# download data
df = yf.download(stock_symbol, start, end)

if df.empty:
    st.error("No data found for the given stock symbol. Please try another one.")
    st.stop()
st.subheader(f"Raw data for {stock_symbol}")
st.write(df.describe())

# just create 2 tab for prediction and visualization
tab1, tab2 = st.tabs(["Prediction", "Visualization"])

with tab1:
    col1, col2 = st.columns(2)


    with col1:
        st.subheader('Close Price vs Time Chart')
        fig = plt.figure(figsize=(12, 6))

        plt.plot(df['Close'], label='Close Price')
        plt.xlabel('Time')
        plt.ylabel('Price')         
        st.pyplot(fig)

    with col2:
        st.subheader('Closing Price vs Time Chart with 100MA')
        ma100 = df.Close.rolling(100).mean()
        fig = plt.figure(figsize=(12, 6))
        plt.plot(ma100, label='MA100')
        plt.plot(df['Close'], label='Close Price')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(fig)

    st.subheader('Closing Price vs Time Chart with 100MA and 200MA')
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize=(12, 6))
    plt.plot(ma100, label='MA100')
    plt.plot(ma200, label='MA200')
    plt.plot(df['Close'], label='Close Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig)

# Splitting data
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

# Load model
try:
    model = load_model('keras_model.h5')
except:
    st.error("Model file 'keras_model.h5' not found. Please make sure it's in the correct directory.")
    st.stop()

# Scaling ONLY on training data
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_scaled = scaler.fit_transform(data_training)

# Prepare test data
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Predictions
y_predicted = model.predict(x_test)
y_predicted = scaler.inverse_transform(y_predicted)  # proper inverse
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

with tab2:
    st.subheader('Future Price Prediction')

    # User input for specific prediction date
    target_date = st.date_input("Select a date for prediction", pd.to_datetime("2025-10-10"))

    last_date = df.index[-1].date()
    days_ahead = (target_date - last_date).days

    if days_ahead <= 0:
        st.error("Selected date must be after the last available date in the dataset.")
    else:
        # Use last 100 days to predic..
        last_100_days = df['Close'].tail(100).values.reshape(-1, 1)
        last_100_days_scaled = scaler.transform(last_100_days)

        future_predictions = []
        current_batch_scaled = last_100_days_scaled.copy()

        for i in range(days_ahead):
            
            current_batch_reshaped = current_batch_scaled.reshape(1, 100, 1)
            
           
            current_pred_scaled = model.predict(current_batch_reshaped)[0]
            future_predictions.append(current_pred_scaled)

            
            current_batch_scaled = np.append(current_batch_scaled[1:], current_pred_scaled)
            current_batch_scaled = current_batch_scaled.reshape(-1, 1)

        
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        # Create future dates
        future_dates = pd.date_range(start=last_date, periods=days_ahead + 1)[1:]

        # Plot
        fig4 = plt.figure(figsize=(12, 6))
        plt.plot(df.index[-100:], df['Close'].tail(100), 'b-', label='Historical Price')
        plt.plot(future_dates, future_predictions, 'r--', label='Future Predictions')
        plt.axvline(target_date, color='g', linestyle='--', label='Target Date')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title(f'{user_input} Stock Price Prediction until {target_date}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig4)

        # Show final prediction 
        st.subheader(f"Predicted Price on {target_date}:")
        st.success(f"${future_predictions[-1][0]:.2f}")


st.sidebar.header('About')
st.sidebar.info(
    """
    This app uses a pre-trained LSTM neural network to predict stock prices.
    The model was trained on historical data and makes predictions based on 
    the last 100 days of closing prices.
    """
)