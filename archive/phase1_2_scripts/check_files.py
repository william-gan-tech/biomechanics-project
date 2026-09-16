import os

base_dir = r'C:\Users\qgan2\OneDrive\Desktop\Research - biomechanics_project\biomechanics-project'
data_dir = os.path.join(base_dir, 'data')

print("Files currently inside your data folder:")
for filename in os.listdir(data_dir):
    print(f" - {filename}")