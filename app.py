import os
os.environ["STREAMLIT_WATCHDOG_MODE"] = "none"

import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

st.set_page_config(page_title="COVID-19 Time Series", layout="wide")
st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #ffe6f0, #e6ccff);
        }
        header, footer {
            visibility: hidden;
        }
    </style>
""", unsafe_allow_html=True)


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
if 'people_vaccinated' in filtered_df.columns:
    st.subheader("💉 Vaccination Trends")
    fig_vax = px.line(filtered_df, x='date', y='people_vaccinated',
                      title="People Vaccinated Over Time",
                      labels={"people_vaccinated": "Vaccinated"})
    st.plotly_chart(fig_vax, use_container_width=True)

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

    st.markdown("### 📥 Download Forecast")
    if st.button("Download Forecast Data"):
        csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False).encode()
        st.download_button("Download Forecast CSV", csv, "forecast.csv", "text/csv")
else:
    st.warning("Not enough data for forecasting.")

if st.button("Download Forecast PDF"):
    # Create a PDF buffer
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    # Title
    elements.append(Paragraph("COVID-19 Forecast (Next 30 Days)", styles['Title']))
    elements.append(Paragraph(f"Country: {selected_country}", styles['Normal']))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d')}", styles['Normal']))
    elements.append(Paragraph(" ", styles['Normal']))

    # Forecast Table Data
    table_data = [["Date", "Predicted Cases", "Lower Bound", "Upper Bound"]]
    for i in range(len(forecast)):
        row = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].iloc[i]
        table_data.append([
            row['ds'].strftime('%Y-%m-%d'),
            f"{row['yhat']:.2f}",
            f"{row['yhat_lower']:.2f}",
            f"{row['yhat_upper']:.2f}"
        ])

    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
    ]))
    elements.append(table)

    # Build PDF
    doc.build(elements)
    buffer.seek(0)

    st.download_button(
        label="Download Forecast PDF",
        data=buffer,
        file_name="covid_forecast.pdf",
        mime="application/pdf"
    )

# Footer
st.markdown("---")
st.markdown("🔬 Created by **Sarmistha Sen** | Data Source: [Our World In Data](https://ourworldindata.org/coronavirus)")
