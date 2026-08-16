import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title='Advanced Biomechanics Dashboard', layout='wide')

st.title('? Biomechanics Fatigue & Cross-Skater Anomaly Dashboard')
st.markdown('### Joint-Specific MSE Decomposition & Cross-Subject Generalization')

# Sidebar controls
st.sidebar.header('Pipeline Controls')
training_skater = st.sidebar.selectbox('Baseline Model Trained On', ['Elite Skater A (Sven Kramer)', 'Skater B'])
eval_skater = st.sidebar.selectbox('Target Evaluation Skater', ['Skater B (Multivariate)', 'Test Skater 1', 'Test Skater 2'])
threshold = st.sidebar.slider('Anomaly Threshold', 0.01, 0.10, 0.045, 0.005)

st.markdown(f'Evaluating cross-generalization: Model trained on **{training_skater}** tested on **{eval_skater}**')

# Simulate multi-joint features
np.random.seed(101)
frames = 35
x_vals = np.arange(frames)

joint_errors = {
    'Knee Flexion': np.random.uniform(0.015, 0.045, frames),
    'Hip Angle': np.random.uniform(0.025, 0.065, frames),
    'Ankle Dorsiflexion': np.random.uniform(0.010, 0.035, frames),
    'Torso Lean': np.random.uniform(0.020, 0.055, frames)
}

overall_score = np.mean(list(joint_errors.values()), axis=0)

col1, col2, col3 = st.columns(3)
col1.metric('Mean Reconstruction Error', f'{np.mean(overall_score):.4f}')
col2.metric('Peak Joint Error', f'{np.max(overall_score):.4f}')
col3.metric('Status', 'Flagged Anomaly' if np.max(overall_score) > threshold else 'Normal Form')

tab1, tab2 = st.tabs(['?? Overall Anomaly Progression', '?? Joint-Specific MSE Decomposition'])

with tab1:
    st.markdown('### Global Reconstruction Error vs Threshold')
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_vals, overall_score, label='Overall MSE Score', color='black', linewidth=2)
    ax.axhline(y=threshold, color='red', linestyle='--', label='Anomaly Threshold')
    ax.set_xlabel('Frame / Window Index')
    ax.set_ylabel('Mean Squared Error')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

with tab2:
    st.markdown('### Isolating Breakdown: Knee vs. Hip vs. Ankle vs. Torso')
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    for joint_name, err_vals in joint_errors.items():
        ax2.plot(x_vals, err_vals, label=joint_name, marker='.')
    ax2.axhline(y=threshold, color='red', linestyle=':', label='Threshold')
    ax2.set_xlabel('Frame / Window Index')
    ax2.set_ylabel('Feature-Level MSE Loss')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True)
    plt.tight_layout()
    st.pyplot(fig2)

st.success('Advanced multi-joint analysis loaded successfully!')
