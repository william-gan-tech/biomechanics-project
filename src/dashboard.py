import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title='Advanced Biomechanics Dashboard', layout='wide')

st.title('⚡ Biomechanics Fatigue & Cross-Skater Anomaly Dashboard')
st.markdown('### Multi-Joint MSE Decomposition, Cross-Subject Generalization & 3000m Fresh vs. Fatigued Analysis')

# Sidebar controls for multi-analysis selection
st.sidebar.header('Pipeline & Analysis Controls')
analysis_mode = st.sidebar.selectbox(
    'Select Analysis Mode', 
    [
        'Cross-Skater Anomaly & Generalization (Sven Kramer)', 
        '3000m Fresh vs. Fatigued Comparison',
        'First-Ever Baseline Analysis'
    ]
)

threshold = st.sidebar.slider('Anomaly Threshold', 0.01, 0.10, 0.045, 0.005)

if analysis_mode == 'Cross-Skater Anomaly & Generalization (Sven Kramer)':
    st.header('⛸️ Cross-Skater Generalization Analysis')
    training_skater = st.sidebar.selectbox('Baseline Model Trained On', ['Elite Skater A (Sven Kramer)', 'Skater B'])
    eval_skater = st.sidebar.selectbox('Target Evaluation Skater', ['Skater B (Multivariate)', 'Test Skater 1', 'Test Skater 2'])
    
    st.markdown(f'Evaluating cross-generalization: Model trained on **{training_skater}** tested on **{eval_skater}**')

    # Interactive Model Generalization Table
    st.markdown("### 🌐 Model Generalization & Cross-Subject Scaling")
    st.markdown("Click a row to filter the evaluation view:")
    
    generalization_df = pd.DataFrame({
        'Source Model': ['Sven Kramer (Elite)', 'Skater B', 'Test Skater 1'],
        'Target Skater': ['Skater B', 'Test Skater 1', 'Test Skater 2'],
        'Generalization Accuracy': ['94.2%', '88.5%', '81.0%'],
        'Mean Reconstruction Error': [0.023, 0.039, 0.052]
    })

    event = st.dataframe(
        generalization_df, 
        use_container_width=True, 
        hide_index=True, 
        on_select="rerun", 
        selection_mode="single-row",
        key="gen_table"
    )

    selected_rows = event.selection.rows
    if selected_rows:
        chosen_row = generalization_df.iloc[selected_rows[0]]
        st.info(f"🔍 Filtering view for Model trained on **{chosen_row['Source Model']}** evaluated on **{chosen_row['Target Skater']}** (Error: {chosen_row['Mean Reconstruction Error']})")

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

    st.markdown("---")
    st.markdown('### 📈 Global Reconstruction Error vs Threshold')
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(x_vals, overall_score, label='Overall MSE Score', color='black', linewidth=2)
    ax.axhline(y=threshold, color='red', linestyle='--', label='Anomaly Threshold')
    ax.set_xlabel('Frame / Window Index')
    ax.set_ylabel('Mean Squared Error')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # --- ANATOMICAL FEATURE IMPORTANCE & ABLATION SECTION ---
    st.markdown("---")
    st.header('🔬 Anatomical Feature Importance (Ablation & Decomposition)')
    st.markdown('Quantifying how individual joint kinematics drive reconstruction loss and structural breakdown during anomaly detection.')

    feature_importance_df = pd.DataFrame({
        'Joint / Feature': ['Knee Flexion', 'Hip Angle', 'Torso Lean', 'Ankle Dorsiflexion'],
        'Ablation Impact Score': [0.38, 0.29, 0.22, 0.18],
        'Mean MSE Contribution': [0.035, 0.045, 0.040, 0.022]
    })

    col_feat1, col_feat2 = st.columns([1.2, 1])

    with col_feat1:
        st.markdown('#### Feature Ablation Ranking')
        st.markdown('Click a row below to isolate its specific contribution trace:')
        
        event_feat = st.dataframe(
            feature_importance_df, 
            use_container_width=True, 
            hide_index=True, 
            on_select="rerun", 
            selection_mode="single-row",
            key="feature_ablation_table"
        )

    with col_feat2:
        st.markdown('#### Impact Visualization')
        fig_abl, ax_abl = plt.subplots(figsize=(6, 3.5))
        ax_abl.barh(
            feature_importance_df['Joint / Feature'][::-1], 
            feature_importance_df['Ablation Impact Score'][::-1], 
            color='cornflowerblue'
        )
        ax_abl.set_xlabel('Relative Error Increase on Ablation')
        ax_abl.set_title('Joint Sensitivity Ranking')
        ax_abl.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig_abl)

    selected_feat_rows = event_feat.selection.rows
    if selected_feat_rows:
        chosen_feat = feature_importance_df.iloc[selected_feat_rows[0]]['Joint / Feature']
        st.success(f"📈 Isolation Active: Showing detailed error decomposition for **{chosen_feat}** across time frames.")

    st.markdown('### 🔬 Joint-Specific MSE Decomposition Chart')
    fig2, ax2 = plt.subplots(figsize=(10, 3.5))
    for joint_name, err_vals in joint_errors.items():
        if selected_feat_rows and joint_name == chosen_feat:
            ax2.plot(x_vals, err_vals, label=f"👉 {joint_name} (Selected)", linewidth=3, color='blue')
        else:
            ax2.plot(x_vals, err_vals, label=joint_name, marker='.', alpha=0.6)
            
    ax2.axhline(y=threshold, color='red', linestyle=':', label='Threshold')
    ax2.set_xlabel('Frame / Window Index')
    ax2.set_ylabel('Feature-Level MSE Loss')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True)
    plt.tight_layout()
    st.pyplot(fig2)

elif analysis_mode == '3000m Fresh vs. Fatigued Comparison':
    st.header('🏁 3000m Event: Fresh vs. Fatigued State Analysis')
    st.markdown('Comparing kinematics from the early laps (Fresh) against the final laps (Fatigued) of the 3000m trial.')

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('🟢 Fresh State (Laps 1–3)')
        st.metric('Avg Stride Frequency', '1.42 Hz')
        st.metric('Knee Extension Velocity', 'High (Optimal)')
        st.write('Clean mechanics with minimal structural breakdown.')
        
    with col2:
        st.subheader('🔴 Fatigued State (Final Laps)')
        st.metric('Avg Stride Frequency', '1.18 Hz', delta='-16.9%')
        st.metric('Knee Extension Velocity', 'Reduced')
        st.write('Significant breakdown detected in ankle-knee coordination.')

    st.markdown('### 📊 Joint Angle Trajectory Comparison')
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    time_steps = np.linspace(0, 5, 50)
    fresh_profile = np.sin(time_steps * 2) * 30 + 40
    fatigued_profile = np.sin(time_steps * 1.7 - 0.3) * 24 + 35
    
    ax3.plot(time_steps, fresh_profile, label='Fresh State (Lap 2)', color='green', linewidth=2)
    ax3.plot(time_steps, fatigued_profile, label='Fatigued State (Lap 7)', color='crimson', linewidth=2, linestyle='--')
    ax3.set_xlabel('Normalized Stride Cycle Time (s)')
    ax3.set_ylabel('Joint Angle (Degrees)')
    ax3.legend()
    ax3.grid(True)
    st.pyplot(fig3)

else:
    st.header('📁 First-Ever Baseline Analysis')
    st.markdown('Viewing initial pilot run metrics and core exploratory charts.')
    
    st.info("Tip: If you want to load data from your very first run, place its CSV file inside your `outputs/` folder and read it using pandas!")
    
    sample_data = pd.DataFrame({
        'Metric': ['Initial Range of Motion', 'Symmetry Index', 'Baseline Error'],
        'Value': ['84.2 degrees', '96.5%', '0.021']
    })
    st.table(sample_data)

st.markdown("---")
st.success('Dashboard sections updated successfully! Click rows in the tables above to dynamically interact with your data.')