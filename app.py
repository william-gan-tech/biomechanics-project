import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title='Biomechanics Dashboard', layout='wide')

st.title('⚡ Biomechanics Fatigue & Anomaly Detection Dashboard')
st.markdown('### Multi-Subject Generalization & Joint-Specific MSE Decomposition')

# Sidebar
st.sidebar.header('Controls')
skater_choice = st.sidebar.selectbox('Select Skater', ['test_skater_1', 'test_skater_2', 'elite_sven_kramer'])
threshold = st.sidebar.slider('Anomaly Threshold', 0.01, 0.10, 0.045, 0.005)

# Main Layout
st.markdown(f'Currently displaying analysis for: **{skater_choice}**')

# Generate sample visual data for the dashboard
np.random.seed(42)
frames = 30
x_vals = np.arange(frames)
knee_err = np.random.uniform(0.02, 0.05, frames)
hip_err = np.random.uniform(0.03, 0.07, frames)
anomaly_score = (knee_err + hip_err) / 2

col1, col2, col3 = st.columns(3)
col1.metric('Mean Anomaly Score', f'{np.mean(anomaly_score):.4f}')
col2.metric('Peak Anomaly', f'{np.max(anomaly_score):.4f}')
col3.metric('Threshold Status', 'Normal' if np.max(anomaly_score) < threshold else 'Alert')

# Plotting
st.markdown('### 📈 Joint Reconstruction Error & Fatigue Progression')
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(x_vals, knee_err, label='Knee Error', marker='o')
ax.plot(x_vals, hip_err, label='Hip Error', marker='^')
ax.plot(x_vals, anomaly_score, label='Anomaly Score', color='black', linestyle='--')
ax.axhline(y=threshold, color='red', linestyle=':', label='Threshold')
ax.set_xlabel('Window Index / Time Progression')
ax.set_ylabel('Reconstruction Error / MSE')
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.success('Dashboard loaded successfully!')
