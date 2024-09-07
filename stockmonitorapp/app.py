from utils.base_stock_reader import snp500_reader
from models.moving_average import moving_average
import datetime as dt
import streamlit as st
import matplotlib.pyplot as plt 
from matplotlib import style
import logging

style.use('ggplot')

@st.cache_data
@st.cache_resource
def fetch_stock_data(source, stock_symbol, start_date, end_date):
    stock_object = snp500_reader()
    return stock_object.get_web_stock_data(source, stock_symbol, start_date, end_date)

@st.cache_data
@st.cache_resource
def fetch_stock_data_period(source, stock_symbol, period):
    stock_object = snp500_reader()
    return stock_object.get_web_stock_data_period(stock_symbol, period)

def main():
    st.title("Stock Monitor")
    st.write("Welcome to the Stock Monitor App!")
    st.write("This app will allow you to track the stock prices of your favorite companies.")
    st.write("Please select the stock you would like to track from the sidebar.")

    stock_list = ["TSLA", "NVDA", "AAPL", "GOOGL", "MSFT"]
    selected_stock = st.sidebar.selectbox("Select a stock", stock_list)

    # Dates do not work on mobile. To fix this, we can use the following code:
    # start_date = st.sidebar.date_input("Start Date", dt.datetime(2019, 1, 1))
    # end_date = st.sidebar.date_input("End Date", dt.datetime.now())
    # data = fetch_stock_data('stooq', selected_stock, start_date, end_date)

    # Add Radio button for selecting the period
    period = st.sidebar.radio("Select the period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"])
    data = fetch_stock_data_period('stooq', selected_stock, period)

    # Add Radio button for selecting an interactive chart
    interactive_chart = st.sidebar.checkbox("Interactive Chart", value=False)

    logging.debug(f"Data: {data}")

    if data is not None and not data.empty:

        st.subheader(f"Stock Data for {selected_stock} for last 5 days")
        st.write(data.tail())

        # Calculate moving averages
        data['50ma'] = moving_average(data['Close'], 50)
        data['200ma'] = moving_average(data['Close'], 200)

        # Calculate moving averages
        data['50ma'] = data['Close'].rolling(window=50).mean()
        data['200ma'] = data['Close'].rolling(window=200).mean()

        # Plot 50-day and 200-day moving averages as well as the closing price
        st.subheader(f"Time Series Plot for {selected_stock}")
        st.write("Golden Cross - 50 day moving average (short term) crosses above the 200 day moving average (long term). This is a bullish signal.")
        st.write("Death Cross - 50 day moving average (short term) crosses below the 200 day moving average (long term)")

        if interactive_chart:
            # Create a DataFrame with the columns to plot
            plot_data = data[['Close', '50ma', '200ma']]
            st.line_chart(plot_data)
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [5, 1]})

            ax1.plot(data.index, data['Close'], label='Close Price')
            ax1.plot(data.index, data['50ma'], label='50 day moving average')
            ax1.plot(data.index, data['200ma'], label='200 day moving average')
            ax1.legend()
            ax1.set_ylabel('Price')

            ax2.bar(data.index, data['Volume'], label='Volume')
            ax2.set_ylabel('Volume')
            ax2.set_xlabel('Date')

            st.pyplot(fig)
    else:
        st.write("No data available for the selected stock and date range.")

if __name__ == "__main__":
    main()