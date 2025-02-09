import os
# Create folder outputs/predictions 
save_folder = os.path.join(".", "outputs", "predictions")
os.makedirs(save_folder, exist_ok=True)
# Create folder outputs/models 
save_folder = os.path.join(".", "outputs", "models")
os.makedirs(save_folder, exist_ok=True)
# Create folder data/processed 
save_folder = os.path.join(".", "data", "processed")
os.makedirs(save_folder, exist_ok=True)