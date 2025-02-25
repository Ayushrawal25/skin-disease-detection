import os

def check_data_structure():
    print("Checking data structure...")
    
    # Check raw data
    raw_path = 'data/raw_data'
    print("\nContents of raw_data:")
    if os.path.exists(raw_path):
        print(os.listdir(raw_path))
        
        # Check for image directories
        for item in os.listdir(raw_path):
            item_path = os.path.join(raw_path, item)
            if os.path.isdir(item_path):
                print(f"\nContents of {item}:")
                print(len(os.listdir(item_path)), "files")
    
    # Check metadata file
    metadata_path = os.path.join(raw_path, 'HAM10000_metadata.csv')
    print("\nMetadata file exists:", os.path.exists(metadata_path))
    
    # Check processed data structure
    processed_path = 'data/processed_data'
    print("\nProcessed data structure:")
    if os.path.exists(processed_path):
        for split in ['train', 'validation', 'test']:
            split_path = os.path.join(processed_path, split)
            if os.path.exists(split_path):
                print(f"\n{split} directory exists")
                print("Classes:", os.listdir(split_path))

if __name__ == "__main__":
    check_data_structure()