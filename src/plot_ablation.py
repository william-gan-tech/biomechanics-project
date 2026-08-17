import matplotlib.pyplot as plt
import numpy as np

def plot_ablation_results():
    # Data from your ablation study
    features = ['Knees Only', 'Hips Only', 'All Features (Combined)']
    losses = [0.6397, 0.5672, 0.5251]
    
    # Set up the plot style
    plt.figure(figsize=(8, 5))
    bars = plt.bar(features, losses, color=['#4C72B0', '#DD8452', '#55A868'], width=0.5)
    
    # Add titles and labels
    plt.title('Kinematic Ablation Study: Model Performance by Joint Feature Set', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Feature Combination', fontsize=10, fontweight='bold')
    plt.ylabel('Final Training Loss (MSE / BCELoss)', fontsize=10, fontweight='bold')
    plt.ylim(0.4, 0.7)
    
    # Add exact value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the plot
    output_path = "comparison_plot.png"
    plt.savefig(output_path, dpi=300)
    print(f"✅ Ablation comparison plot successfully saved to '{output_path}'!")
    plt.show()

if __name__ == "__main__":
    plot_ablation_results()