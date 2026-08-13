import streamlit as st
import pandas as pd
import os

# Page configuration
st.set_page_config(page_title="Biomechanics Fatigue Dashboard", layout="wide")

st.title("⚡ Speed Skating Biomechanics & Fatigue Dashboard")
st.write("Analyzing form degradation and stride anomaly scores using autoencoder reconstruction loss.")

# 1. Load Data Safely
csv_path = "fatigue_results.csv"

if not os.path.exists(csv_path):
    st.error(f"⚠️ Missing file: `{csv_path}` was not found in the root directory.")
    st.info("Run your model script in your terminal first by typing: `python main.py`")
else:
    df = pd.read_csv(csv_path)

    # 2. Sidebar controls
    st.sidebar.header("Dashboard Controls")
    threshold = st.sidebar.slider(
        "Fatigue Anomaly Threshold", 
        float(df["Anomaly_Score"].min()), 
        float(df["Anomaly_Score"].max()), 
        float(df["Anomaly_Score"].mean())
    )

    # 3. Key Metrics View
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Windows Analyzed", len(df))
    col2.metric("Peak Anomaly Score", f"{df['Anomaly_Score'].max():.4f}")
    col3.metric("Baseline Error (Start)", f"{df.iloc[0]['Anomaly_Score']:.4f}")

    # 4. Main Chart
    st.subheader("📈 Stride Window vs. Reconstruction Error (Fatigue Trend)")
    st.line_chart(df.set_index("Window_Index")["Anomaly_Score"])

    # 5. Data Table View
    with st.expander("🔍 View Raw Analysis Data Table"):
        st.dataframe(df)