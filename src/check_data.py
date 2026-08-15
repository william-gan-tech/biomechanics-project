import pandas as pd

# Load your new multivariate CSV file
df = pd.read_csv("extracted_multivariate_angles.csv")

# Print the first 5 rows to check columns and values
print("--- First 5 Rows of Data ---")
print(df.head())

# Print column names to confirm all joints are present
print("\n--- Tracked Columns ---")
print(df.columns.tolist())