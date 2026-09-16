import os
import subprocess
import sys

def run_pipeline():
    print("====================================================")
    print("🚀 Starting Biomechanics Research Pipeline")
    print("====================================================")
    
    scripts = [
        "src/demo_skater_a.py",
        "src/lead_time_experiment.py",
        "src/stride_fatigue_report.py",
        "src/cross_validate.py"
    ]
    
    for script in scripts:
        if os.path.exists(script):
            print(f"\nRunning: {script}...")
            # Use subprocess to run the script
            # We use sys.executable to ensure we use the same Python interpreter
            subprocess.run([sys.executable, script], check=True)
        else:
            print(f"Warning: {script} not found, skipping...")
            
    print("\n====================================================")
    print("✅ Pipeline Complete! Check the 'outputs/' folder for results.")
    print("====================================================")

if __name__ == "__main__":
    run_pipeline()