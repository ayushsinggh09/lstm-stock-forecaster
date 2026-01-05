Stock Prediction App
This is a small project where I use Streamlit, TensorFlow/Keras, and Yahoo Finance data to predict stock prices.
The model is an LSTM network trained on historical closing prices. The app lets you enter any stock ticker (like KOTAKBANK.NS or TATAMOTORS.NS) and see both past trends and future predictions.

How to run
Clone the repo:

bash
git clone https://github.com/ayushsinggh09/lstm-stock-forecaster.git
cd lstm-stock-forecaster


Install requirements:
pip install -r requirements.txt

Start the app:
streamlit run app.py

What you get
Charts of closing prices with moving averages (100MA, 200MA).
A trained LSTM model (keras_model.h5) that predicts future prices.
Option to select a date and see the predicted price for that day.

Notes
This is for learning purposes only, not financial advice.
Make sure the model file (keras_model.h5) is in the project folder before running.


