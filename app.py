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
st.markdown("Advanced temporal trajectory analysis with joint-specific error decomposition for tracking form breakdown.")

# 2. Load Data Safely
csv_path = "fatigue_results.csv"

if not os.path.exists(csv_path):
    st.error(f"⚠️ Missing file: `{csv_path}` was not found in the root directory.")
    st.info("Run your model script in your terminal first by typing: `python main.py`")
else:
    df = pd.read_csv(csv_path)

    # Determine available analytical columns for joint selection
    available_columns = [col for col in df.columns if col != "Window_Index"]
    default_metric = "Anomaly_Score" if "Anomaly_Score" in available_columns else available_columns[0]

    # 3. Sidebar Advanced Controls & Joint Toggle
    st.sidebar.header("⚙️ Dashboard Controls")
    
    if len(available_columns) > 1:
        selected_joint = st.sidebar.selectbox(
            "Select Anatomical Joint Region", 
            available_columns
        )
    else:
        selected_joint = default_metric

    threshold = st.sidebar.slider(
        f"Fatigue Threshold ({selected_joint})", 
        float(df[selected_joint].min()), 
        float(df[selected_joint].max()), 
        float(df[selected_joint].mean())
    )

    # 4. Dynamic Warning Alert System
    max_score = df[selected_joint].max()
    if max_score > threshold:
        st.warning(f"🚨 **Fatigue Alert:** Peak score for **{selected_joint.replace('_', ' ')}** ({max_score:.4f}) exceeded your threshold!")
    else:
        st.success(f"✅ **Status Normal:** {selected_joint.replace('_', ' ')} trajectory remains within baseline parameters.")

    # 5. Executive Metrics Cards
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Stride Windows", len(df))
    col2.metric(f"Peak {selected_joint.replace('_', ' ')}", f"{max_score:.4f}")
    col3.metric("Baseline Error (Start)", f"{df.iloc[0][selected_joint]:.4f}")

    # 6. Main Interactive Time-Series Chart
    st.subheader(f"📈 Stride Window vs. Error Trend: {selected_joint.replace('_', ' ')}")
    st.line_chart(df.set_index("Window_Index")[selected_joint])

    # 7. Downloadable Summary Report
    st.subheader("📥 Export Data")
    csv_data = df.to_csv(index=False)
    st.download_button(
        label="Download Full Biomechanical Report (CSV)",
        data=csv_data,
        file_name="biomechanics_fatigue_report.csv",
        mime="text/csv"
    )

    # 8. Raw Data Expander Table
    with st.expander("🔍 View Raw Multi-Joint Analysis Table"):
        st.dataframe(df)