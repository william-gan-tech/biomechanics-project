import os
import pandas as pd

print("Current Working Directory:", os.getcwd())
print("Files in current directory:", os.listdir('.'))

data_folder_path = os.path.join(os.getcwd(), 'data')
if os.path.exists(data_folder_path):
    print("Files inside 'data' folder:", os.listdir('data'))
else:
    print("ERROR: 'data' folder not found by Python!")