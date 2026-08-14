import streamlit as st
import pandas as pd
import os

# 1. Page Configuration
st.set_page_config(
    page_title="Biomechanics Fatigue Dashboard",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Speed Skating Biomechanics & Fatigue Dashboard")
st.markdown("Advanced temporal trajectory analysis for tracking form breakdown and predicting athletic fatigue.")

# 2. Load Data Safely
csv_path = "fatigue_results.csv"

if not os.path.exists(csv_path):
    st.error(f"⚠️ Missing file: `{csv_path}` was not found in the root directory.")
    st.info("Run your model script in your terminal first by typing: `python main.py`")
else:
    df = pd.read_csv(csv_path)

    # 3. Sidebar Advanced Controls
    st.sidebar.header("⚙️ Dashboard Controls")
    threshold = st.sidebar.slider(
        "Fatigue Anomaly Threshold", 
        float(df["Anomaly_Score"].min()), 
        float(df["Anomaly_Score"].max()), 
        float(df["Anomaly_Score"].mean())
    )

    # 4. Dynamic Warning Alert System
    max_score = df["Anomaly_Score"].max()
    if max_score > threshold:
        st.warning(f"🚨 **Fatigue Alert:** Peak anomaly score ({max_score:.4f}) exceeded your safety threshold!")
    else:
        st.success("✅ **Status Normal:** Movement trajectory remains within baseline parameters.")

    # 5. Executive Metrics Cards
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Stride Windows", len(df))
    col2.metric("Peak Anomaly Score", f"{max_score:.4f}")
    col3.metric("Baseline Error (Start)", f"{df.iloc[0]['Anomaly_Score']:.4f}")

    # 6. Main Interactive Time-Series Chart
    st.subheader("📈 Stride Window vs. Reconstruction Error (Fatigue Trend)")
    st.line_chart(df.set_index("Window_Index")["Anomaly_Score"])

    # 7. Downloadable Summary Report
    st.subheader("📥 Export Data")
    csv_data = df.to_csv(index=False)
    st.download_button(
        label="Download Fatigue Report (CSV)",
        data=csv_data,
        file_name="biomechanics_fatigue_report.csv",
        mime="text/csv"
    )

    # 8. Raw Data Expander Table
    with st.expander("🔍 View Raw Analysis Data Table"):
        st.dataframe(df)