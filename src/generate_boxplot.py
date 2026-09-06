import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_generalization_boxplot():
    # Load the dual-pass evaluation CSV logs
    raw_df = pd.read_csv("data/generalization_raw.csv")
    norm_df = pd.read_csv("data/generalization_normalized.csv")
    
    # Tag the source for comparative plotting
    raw_df['Normalization'] = 'Raw (Unscaled)'
    norm_df['Normalization'] = 'Bone-Normalized (Phase 3)'
    
    # Combine datasets
    combined_df = pd.concat([raw_df, norm_df])
    
    # Set up the visualization style
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")
    
    # Generate side-by-side boxplots
    ax = sns.boxplot(
        x='Normalization', 
        y='mean_reconstruction_mse', 
        data=combined_df, 
        palette="Set2",
        showmeans=True
    )
    
    plt.title("Cross-Subject Generalization: Raw vs. Bone-Normalized MSE", fontsize=12, fontweight='bold', pad=15)
    plt.ylabel("Mean Reconstruction MSE (Log Scale)", fontsize=10)
    plt.xlabel("Pipeline Configuration", fontsize=10)
    
    # Use log scale on Y-axis due to the massive scale difference (~4500 vs ~0.5)
    ax.set_yscale('log')
    
    # Save the figure for research reporting
    output_path = "data/generalization_boxplot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Publication-ready boxplot successfully saved to {output_path}")

if __name__ == "__main__":
    generate_generalization_boxplot()