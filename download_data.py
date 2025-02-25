import os
import kaggle
from tqdm import tqdm

def download_dataset():
    print("Downloading HAM10000 dataset...")
    
    # Create directories if they don't exist
    os.makedirs('data/raw_data', exist_ok=True)
    
    try:
        # Download dataset
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            'kmader/skin-cancer-mnist-ham10000',
            path='data/raw_data',
            unzip=True
        )
        print("Download completed successfully!")
        
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")

if __name__ == "__main__":
    download_dataset()