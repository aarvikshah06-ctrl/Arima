import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

st.set_page_config(
    page_title="Indian Stock ARIMA Forecast",
    layout="wide"
)

st.title("📈 ARIMA Forecasting for Indian Stocks")

ticker = st.text_input(
    "Enter NSE Stock Symbol",
    value="RELIANCE.NS"
)

forecast_months = st.slider(
    "Forecast Months Ahead",
    1,
    24,
    12
)

if st.button("Generate Forecast"):

    try:

        st.info("Downloading last 5 years data...")

        data = yf.download(
            ticker,
            period="5y",
            auto_adjust=True
        )

        if len(data) == 0:
            st.error("No data found.")
            st.stop()

        close = data["Close"]

        st.subheader("Historical Data")

        st.line_chart(close)

        st.info("Training ARIMA Model...")

        model = ARIMA(
            close,
            order=(5,1,0)
        )

        model_fit = model.fit()

        forecast = model_fit.forecast(
            steps=forecast_months
        )

        future_dates = pd.date_range(
            start=close.index[-1],
            periods=forecast_months + 1,
            freq="M"
        )[1:]

        forecast_df = pd.DataFrame(
            {
                "Date": future_dates,
                "Forecast Price": forecast.values
            }
        )

        st.subheader("Forecast Values")

        st.dataframe(
            forecast_df,
            use_container_width=True
        )

        fig, ax = plt.subplots(figsize=(12,6))

        ax.plot(
            close.index,
            close,
            label="Historical"
        )

        ax.plot(
            future_dates,
            forecast,
            label="Forecast"
        )

        ax.set_title(
            f"{ticker} ARIMA Forecast"
        )

        ax.legend()

        st.pyplot(fig)

        june_2027 = forecast_df[
            (forecast_df["Date"].dt.month == 6)
            &
            (forecast_df["Date"].dt.year == 2027)
        ]

        if len(june_2027) > 0:

            st.success(
                f"Forecasted June 2027 Price: ₹{june_2027.iloc[0]['Forecast Price']:.2f}"
            )

        else:

            st.warning(
                "Extend forecast horizon to include June 2027."
            )

    except Exception as e:

        st.error(str(e))
