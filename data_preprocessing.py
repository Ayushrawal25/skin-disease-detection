import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataPreprocessor:
    def __init__(self):
        self.raw_data_path = 'data/raw_data'
        self.processed_data_path = 'data/processed_data'
        self.image_size = (224, 224)  # Standard size for MobileNetV2
        
        # Create directories for each category
        self.categories = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        self.category_names = {
            'akiec': 'Actinic Keratoses',
            'bcc': 'Basal Cell Carcinoma',
            'bkl': 'Benign Keratosis',
            'df': 'Dermatofibroma',
            'mel': 'Melanoma',
            'nv': 'Melanocytic Nevi',
            'vasc': 'Vascular Lesions'
        }
        self.create_directories()

    def create_directories(self):
        """Create necessary directories for processed data"""
        print("Creating directories...")
        for split in ['train', 'validation', 'test']:
            for category in self.categories:
                os.makedirs(os.path.join(self.processed_data_path, split, category), exist_ok=True)

    def verify_dataset(self):
        """Verify the existence of required files and directories"""
        print("Verifying dataset structure...")
        
        # Check metadata file
        metadata_path = os.path.join(self.raw_data_path, 'metadata', 'HAM10000_metadata.csv')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        print("✓ Found metadata file")

        # Check image directories
        part1_path = os.path.join(self.raw_data_path, 'images', 'HAM10000_images_part_1')
        part2_path = os.path.join(self.raw_data_path, 'images', 'HAM10000_images_part_2')
        
        if not os.path.exists(part1_path) and not os.path.exists(part2_path):
            raise FileNotFoundError("Neither part_1 nor part_2 image directories found")

        # Count images
        total_images = 0
        if os.path.exists(part1_path):
            part1_images = len([f for f in os.listdir(part1_path) if f.endswith('.jpg')])
            total_images += part1_images
            print(f"✓ Found {part1_images} images in part_1")
        
        if os.path.exists(part2_path):
            part2_images = len([f for f in os.listdir(part2_path) if f.endswith('.jpg')])
            total_images += part2_images
            print(f"✓ Found {part2_images} images in part_2")

        print(f"Total images found: {total_images}")
        return total_images > 0

    def process_data(self):
        """Process and split the dataset"""
        try:
            if not self.verify_dataset():
                print("Error: No images found in dataset")
                return 0, 0, 0

            print("\nReading metadata...")
            metadata_path = os.path.join(self.raw_data_path, 'metadata', 'HAM10000_metadata.csv')
            metadata = pd.read_csv(metadata_path)
            
            print(f"Total images in metadata: {len(metadata)}")
            print("\nClass distribution:")
            print(metadata['dx'].value_counts())

            print("\nSplitting dataset...")
            train_df, temp_df = train_test_split(metadata, test_size=0.3, random_state=42, stratify=metadata['dx'])
            val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['dx'])

            print("\nProcessing and copying images...")
            print("Processing training set...")
            self.process_split(train_df, 'train')
            print("Processing validation set...")
            self.process_split(val_df, 'validation')
            print("Processing test set...")
            self.process_split(test_df, 'test')

            return len(train_df), len(val_df), len(test_df)
            
        except Exception as e:
            print(f"Error in process_data: {str(e)}")
            return 0, 0, 0

    def process_split(self, df, split_name):
        """Process and copy images for a specific split"""
        processed_count = 0
        error_count = 0
        
        for idx, row in df.iterrows():
            # Check both part_1 and part_2 folders
            possible_paths = [
                os.path.join(self.raw_data_path, 'images', 'HAM10000_images_part_1', f"{row['image_id']}.jpg"),
                os.path.join(self.raw_data_path, 'images', 'HAM10000_images_part_2', f"{row['image_id']}.jpg")
            ]
            
            # Find the correct path
            src_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    src_path = path
                    break
            
            if src_path is None:
                error_count += 1
                continue
                
            # Destination path
            dst_path = os.path.join(self.processed_data_path, split_name, row['dx'], f"{row['image_id']}.jpg")
            
            try:
                # Read image
                img = cv2.imread(src_path)
                if img is not None:
                    # Resize image
                    img = cv2.resize(img, self.image_size)
                    # Save processed image
                    cv2.imwrite(dst_path, img)
                    processed_count += 1
                    
                    # Show progress
                    if processed_count % 100 == 0:
                        print(f"Processed {processed_count} images in {split_name} set")
                else:
                    error_count += 1
            except Exception as e:
                error_count += 1
                print(f"Error processing image {row['image_id']}: {str(e)}")

        print(f"\nCompleted {split_name} set:")
        print(f"Successfully processed: {processed_count} images")
        if error_count > 0:
            print(f"Errors encountered: {error_count} images")

    def get_data_stats(self):
        """Get statistics about the processed dataset"""
        stats = {'train': {}, 'validation': {}, 'test': {}}
        
        for split in stats.keys():
            for category in self.categories:
                path = os.path.join(self.processed_data_path, split, category)
                stats[split][category] = len(os.listdir(path))
                
        return stats

class DataVisualizer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.categories = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        self.category_names = {
            'akiec': 'Actinic Keratoses',
            'bcc': 'Basal Cell Carcinoma',
            'bkl': 'Benign Keratosis',
            'df': 'Dermatofibroma',
            'mel': 'Melanoma',
            'nv': 'Melanocytic Nevi',
            'vasc': 'Vascular Lesions'
        }

    def plot_data_distribution(self):
        """Plot distribution of images across categories"""
        plt.figure(figsize=(12, 6))
        
        # Collect counts for each category
        counts = {}
        for category in self.categories:
            path = os.path.join(self.data_path, 'train', category)
            counts[self.category_names[category]] = len(os.listdir(path))
        
        # Create bar plot
        sns.barplot(x=list(counts.keys()), y=list(counts.values()))
        plt.xticks(rotation=45, ha='right')
        plt.title('Distribution of Skin Disease Categories in Training Set')
        plt.xlabel('Disease Category')
        plt.ylabel('Number of Images')
        plt.tight_layout()
        plt.show()

    def show_sample_images(self, num_samples=5):
        """Display sample images from each category"""
        plt.figure(figsize=(15, 10))
        
        for idx, category in enumerate(self.categories):
            path = os.path.join(self.data_path, 'train', category)
            images = os.listdir(path)
            
            if not images:
                continue
                
            # Get random samples
            samples = random.sample(images, min(num_samples, len(images)))
            
            for i, image_name in enumerate(samples):
                img_path = os.path.join(path, image_name)
                img = Image.open(img_path)
                
                plt.subplot(len(self.categories), num_samples, idx * num_samples + i + 1)
                plt.imshow(img)
                plt.axis('off')
                if i == 0:
                    plt.ylabel(self.category_names[category], rotation=45, ha='right')
        
        plt.suptitle('Sample Images from Each Category')
        plt.tight_layout()
        plt.show()

def main():
    try:
        print("Starting data preprocessing...")
        preprocessor = DataPreprocessor()
        
        train_count, val_count, test_count = preprocessor.process_data()
        
        if train_count > 0:
            print(f"\nData split complete:")
            print(f"Training samples: {train_count}")
            print(f"Validation samples: {val_count}")
            print(f"Test samples: {test_count}")
            
            print("\nDetailed statistics:")
            stats = preprocessor.get_data_stats()
            for split, categories in stats.items():
                print(f"\n{split.capitalize()} set:")
                for category, count in categories.items():
                    print(f"{category}: {count} images")

            # Initialize visualizer
            visualizer = DataVisualizer('data/processed_data')
            
            print("\nGenerating visualizations...")
            
            # Plot distribution of diseases
            visualizer.plot_data_distribution()
            
            # Show sample images
            visualizer.show_sample_images()
            
            print("\nPreprocessing completed successfully!")
        else:
            print("\nError: Data preprocessing failed. Please check the dataset and try again.")
            
    except Exception as e:
        print(f"\nError during preprocessing: {str(e)}")
        print("Please ensure that:")
        print("1. The dataset is properly organized")
        print("2. All required packages are installed")
        print("3. You have sufficient disk space")

if __name__ == "__main__":
    main()