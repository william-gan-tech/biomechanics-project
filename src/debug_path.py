import os

print("Current Working Directory:", os.getcwd())
base_dir = r'C:\Users\qgan2\OneDrive\Desktop\Research - biomechanics_project\biomechanics-project'
data_dir = os.path.join(base_dir, 'data')
print("Absolute path to data dir:", data_dir)
print("Does data dir exist?", os.path.exists(data_dir))
if os.path.exists(data_dir):
    print("Files found in data dir:", os.listdir(data_dir))