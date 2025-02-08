import os
import requests
import zipfile
import shutil

dataset_urls = {
    "train_data.csv": "https://static.cdn.epam.com/uploads/583f9e4a37492715074c531dbd5abad2/ds/final_project_train_dataset.zip",
    "test_data.csv": "https://static.cdn.epam.com/uploads/583f9e4a37492715074c531dbd5abad2/ds/final_project_test_dataset.zip"
}

save_folder = os.path.join("..", "data", "raw")
os.makedirs(save_folder, exist_ok=True)

def download_zip(url, file_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded: {file_path}")
    except Exception as e:
        print(f"Error downloading: {e}")

def extract_and_save(zip_file, save_folder):
    if not os.path.exists(zip_file):
        raise FileNotFoundError(f"ZIP file not found: {zip_file}")

    extract_folder = os.path.join(save_folder, "temp_extracted")
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        for root, _, files in os.walk(extract_folder):
            for file in files:
                if file.endswith(".csv"):
                    shutil.move(os.path.join(root, file), os.path.join(save_folder, file))
                    print(f"Saved: {file}")
        shutil.rmtree(extract_folder)
    except Exception as e:
        print(f"Error extracting: {e}")

for name, url in dataset_urls.items():
    zip_path = os.path.join(save_folder, f"{name}.zip")
    download_zip(url, zip_path)
    extract_and_save(zip_path, save_folder)
    os.remove(zip_path)

print("All data files saved successfully in '../data/raw/'.")
