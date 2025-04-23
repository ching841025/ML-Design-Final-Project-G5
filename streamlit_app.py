import streamlit as st
import pandas as pd
import requests

# FastAPI backend URL
API_URL = "http://localhost:5000/predict"

st.title("📈 Top 25 Stock Predictor")

st.markdown("Upload a CSV with input features and stock symbols:")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("🔍 Preview of Uploaded Data:")
    st.dataframe(df)

    if st.button("Get Predictions 🚀"):
        try:
            # Convert dataframe to list of dictionaries for JSON
            input_data = df.to_dict(orient="records")
            response = requests.post(API_URL, json=input_data)

            if response.status_code == 200:
                top_25 = response.json()["top_25"]
                
                # Extract the date from the uploaded file (assumes a 'date' column exists)
                try:
                    invest_date = pd.to_datetime(df["date"]).max().strftime("%Y-%m-%d")
                except:
                    invest_date = "Unknown date"
                
                st.success(f"✅ Top 25 stocks selected for investment on {invest_date}")
                st.write(pd.DataFrame(top_25))


            # if response.status_code == 200:
            #     top_25 = response.json()["top_25"]
            #     st.success("✅ Top 25 stocks selected for investing!")
            #     st.write(pd.DataFrame(top_25))
            else:
                st.error(f"❌ Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"⚠️ Exception: {e}")
