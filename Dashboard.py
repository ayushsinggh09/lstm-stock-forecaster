import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinnance as yf

from sklearn.preprocessing import MinMaxScaler
from tensorfow.keras.models import load_model

st.set_page_config(page_title="Stock Price Prediction Dashboard", layout="wide")
st.title("Stock Price Prediction Dashboard")

start = '2015-01-01'
end = '2025-01-01'

st.sidebar.header("User Input Features")
def user_input_features():
    stock_symbol = st.sidebar.text_input("Stock Symbol", "AAPL")
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