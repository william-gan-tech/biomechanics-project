import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def cluster_motion_phases():
    """
    Uses K-Means clustering on sliding window features to automatically 
    segment motion phases and detect anomalous breakdown without relying 
    on fixed peak heights.
    """
    results_path = os.path.join("outputs", "fatigue_results.csv")
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found. Run your pipeline first.")
        return

    df = pd.read_csv(results_path)
    
    # 1. Prepare feature space (e.g., Anomaly Score and Hip Velocity combined)
    if 'Anomaly_Score' not in df.columns or 'Hip_Velocity' not in df.columns:
        print("Required columns missing from CSV.")
        return

    features = df[['Anomaly_Score', 'Hip_Velocity']].values

    # 2. Apply K-Means Clustering to group data into 3 distinct motion states 
    # (e.g., Normal Push, Normal Recovery, Fatigued/Anomalous Form)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(features)
    
    df['Motion_Cluster'] = clusters

    print("=== 🤖 Machine Learning Motion Clustering ===")
    print(f"Total windows clustered: {len(df)}")
    
    # Analyze clusters to find which one represents fatigue/breakdown 
    # (The cluster with the highest average Anomaly Score)
    cluster_means = df.groupby('Motion_Cluster')['Anomaly_Score'].mean()
    fatigue_cluster = cluster_means.idxmax()
    
    print(f"Identified Fatigue Cluster ID: {fatigue_cluster} (Avg Score: {cluster_means[fatigue_cluster]:.4f})")
    
    # Find when this cluster first appears persistently
    fatigue_indices = df[df['Motion_Cluster'] == fatigue_cluster].index
    if len(fatigue_indices) > 0:
        first_occurrence = fatigue_indices[0]
        print(f"⚠️ Machine Learning model flagged structural breakdown at frame: {first_occurrence}")
    
    # Save clustered output
    clustered_output_path = os.path.join("outputs", "clustered_results.csv")
    df.to_csv(clustered_output_path, index=False)
    print(f"Clustered results saved to: {clustered_output_path}")

if __name__ == "__main__":
    cluster_motion_phases()