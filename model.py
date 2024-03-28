import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
st.title("Stock Price Prediction App")
st.write("This app allows you to predict stock prices using the Prophet library.")

START = st.date_input("Select start date",date(2014,1,1))
TODAY = date.today().strftime("%Y-%m-%d")

selected_stock = st.text_input("Enter stock ticker (e.g., AAPL, GOOG, MSFT, GME)", "AAPL")
def validate_ticker(ticker):
    try:
        yf.Ticker(ticker).info  # Try to retrieve info for the ticker
        return True
    except:
        return False

if not validate_ticker(selected_stock):
    st.error("Invalid stock ticker. Please enter a valid stock ticker.")
    st.stop()
    
n_years = st.slider("Years of Prediction:",1,4)
period = n_years*365
@st.cache_data
def load_data(ticker):
    data=yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True)
    return data
data_load_state=st.text("Load Data...")
data=load_data(selected_stock)
data_load_state.text("Loading Data.... Done!")

st.subheader('Raw Data')
st.write(data.tail())
def plot_raw_data():
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='stock_open'))
    
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name='stock_close'))
    fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
plot_raw_data()

#Forecasting
df_train=data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
m=Prophet()
m.fit(df_train)
future=m.make_future_dataframe(periods=period)
forecast=m.predict(future)

#Displaying forecasted data
st.subheader('Forecast Data')
st.write(future.tail())

st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('Forecast components')
fig2= m.plot_components(forecast)
st.write(fig2)








