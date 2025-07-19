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


st.markdown(
    "<h1 style='text-align: center; color: #2E86C1;'>🦠 COVIDlytics</h1>"
    "<h4 style='text-align: center; color: gray;'>Visualize. Analyze. Forecast.</h4>"
    "<h4 style='text-align: center; color: gray;'>Your AI-Powered COVID Insights Hub.</h4>",
    unsafe_allow_html=True
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("https://covid.ourworldindata.org/data/owid-covid-data.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()

countries = df['location'].unique()

# Use expander instead of sidebar
with st.expander("🔽 Filter Options", expanded=True):
    selected_country = st.selectbox("Select Country", sorted(countries))
    start_date = st.date_input("Start Date", datetime(2020, 1, 1))
    end_date = st.date_input("End Date", datetime.now())

# Filter data
country_df = df[df['location'] == selected_country]
filtered_df = country_df[
    (country_df['date'] >= pd.to_datetime(start_date)) &
    (country_df['date'] <= pd.to_datetime(end_date))
]

view_options = ["Overview","Vaccination Trends","Latest Summary Table","Forecasting","Compare Two Countries","Ask the AI Assistant"]
view = st.selectbox("🔍 Select a Feature", view_options)


# Overview
if view == "Overview":
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


# 💉 Vaccination Trends
elif view == "Vaccination Trends":
    if 'people_vaccinated' in filtered_df.columns:
        vax_data = filtered_df[['date', 'people_vaccinated']].dropna()
        if not vax_data.empty:
            st.subheader("💉 Vaccination Trends")
            fig_vax = px.line(vax_data, x='date', y='people_vaccinated',
                              title="People Vaccinated Over Time",
                              labels={"people_vaccinated": "Vaccinated"})
            st.plotly_chart(fig_vax, use_container_width=True)
        else:
            st.info("No vaccination data available for this country in the selected date range")
            

# 📊 Show Latest Summary Table
elif view == "Latest Summary Table":
    st.subheader("📊 Latest Summary Table")

    latest_date = filtered_df['date'].max()
    latest_data = df[(df['location'] == selected_country) & (df['date'] == latest_date)]

    if not latest_data.empty:
        display_cols = ['total_cases', 'new_cases', 'total_deaths', 'new_deaths', 'people_vaccinated']
        display_data = latest_data[display_cols].transpose().reset_index()
        display_data.columns = ['Metric', 'Value']
        st.dataframe(display_data)
    else:
        st.warning("No summary data available for the selected country and date.")


# 📅 Forecasting
elif view == "Forecasting":
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


# 🌍 Compare Two Countries
elif view == "Compare Two Countries":
    st.subheader("🌍 Compare Two Countries")

    col1, col2 = st.columns(2)
    with col1:
        country1 = st.selectbox("Country 1", sorted(df['location'].unique()), key='country1')
    with col2:
        country2 = st.selectbox("Country 2", sorted(df['location'].unique()), key='country2')

    compare_df = df[(df['location'].isin([country1, country2])) & 
                    (df['date'] >= pd.to_datetime(start_date)) & 
                    (df['date'] <= pd.to_datetime(end_date))]

    compare_df = compare_df[['date', 'location', 'new_cases']].dropna(subset=['new_cases'])

    if compare_df['location'].nunique() < 2:
        st.warning("One of the selected countries has insufficient data for comparison.")
    else:
        fig_compare = px.line(compare_df, x='date', y='new_cases', color='location',
                              title="New Cases Comparison Between Two Countries",
                              labels={"new_cases": "New Cases", "location": "Country"})
        st.plotly_chart(fig_compare, use_container_width=True)


# 🧠 AI Chat Assistant
elif view == "Ask the AI Assistant":
    st.subheader("🧠 Ask the AI Assistant")

    # Sample questions
    sample_questions = [
        "",
        "What was the peak number of new cases in the India?",
        "Predict the trend of COVID-19 cases in Japan for next month.",
        "How did vaccinations impact death rates in Italy?",
        "Which country had the lowest number of cases in 2022?",
        "Compare case trends between India and Brazil."
    ]

    # Dropdown to select a sample question
    selected_sample = st.selectbox("💡 Choose a sample question (optional):", sample_questions)

    # If a sample is selected, auto-fill it into the text area (but allow editing)
    user_question = st.text_area("Ask anything about COVID-19 data, trends, or predictions:", value=selected_sample)

    if st.button("Search"):
        if user_question.strip() != "":
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

            prompt = f"""You are a helpful assistant analyzing COVID-19 time series data. 
            The user asked: "{user_question}". 
            Provide an informative, data-backed answer within 150 words."""

            with st.spinner("Thinking..."):
                try:
                    from openai import OpenAI

                    client = OpenAI(api_key=openai.api_key)

                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7,
                        max_tokens=300,
                    )

                    st.success(response.choices[0].message.content.strip())

                except Exception as e:
                    st.error(f"Error from AI assistant: {e}")
        else:
            st.warning("Please enter a question before searching.")



# Footer
st.markdown("---")
st.markdown("🔬 Created by **Sarmistha Sen** | Data Source: [Our World In Data](https://ourworldindata.org/coronavirus)")
