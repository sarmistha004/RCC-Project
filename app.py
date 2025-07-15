import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime

st.set_page_config(page_title="COVID-19 Time Series", layout="wide")

st.title("🦠 COVID-19 Time Series Analysis & Forecasting")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("https://covid.ourworldindata.org/data/owid-covid-data.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()

# Sidebar filters
countries = df['location'].unique()
selected_country = st.sidebar.selectbox("Select Country", sorted(countries))
start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.now())

# Filter data
country_df = df[df['location'] == selected_country]
filtered_df = country_df[(country_df['date'] >= pd.to_datetime(start_date)) & (country_df['date'] <= pd.to_datetime(end_date))]

st.subheader(f"📊 COVID-19 Data for {selected_country}")
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Cases", int(filtered_df['total_cases'].max() or 0))
with col2:
    st.metric("Total Deaths", int(filtered_df['total_deaths'].max() or 0))

# Time series plots
fig_cases = px.line(filtered_df, x='date', y='new_cases', title="Daily New Cases", labels={"new_cases": "New Cases"})
fig_deaths = px.line(filtered_df, x='date', y='new_deaths', title="Daily New Deaths", labels={"new_deaths": "New Deaths"})

st.plotly_chart(fig_cases, use_container_width=True)
st.plotly_chart(fig_deaths, use_container_width=True)

# Forecasting
st.subheader("📅 Forecasting with Prophet")
prophet_data = filtered_df[['date', 'new_cases']].rename(columns={'date': 'ds', 'new_cases': 'y'}).dropna()

if not prophet_data.empty:
    m = Prophet(daily_seasonality=True)
    m.fit(prophet_data)
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)

    fig_forecast = plot_plotly(m, forecast)
    st.plotly_chart(fig_forecast, use_container_width=True)

    if st.button("Download Forecast Data"):
        csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False).encode()
        st.download_button("Download Forecast CSV", csv, "forecast.csv", "text/csv")
else:
    st.warning("Not enough data for forecasting.")

# Footer
st.markdown("---")
st.markdown("🔬 Created by **Sarmistha Sen** | Data Source: [Our World In Data](https://ourworldindata.org/coronavirus)")
