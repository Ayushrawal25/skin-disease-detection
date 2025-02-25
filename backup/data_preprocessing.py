import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil
from tqdm import tqdm

class DataPreprocessor:
    def __init__(self):
        self.raw_data_path = 'data/raw_data'
        self.processed_data_path = 'data/processed_data'
        self.image_size = (224, 224)
        
        # Class information
        self.class_names = {
            'akiec': 'Actinic Keratoses',
            'bcc': 'Basal Cell Carcinoma',
            'bkl': 'Benign Keratosis',
            'df': 'Dermatofibroma',
            'mel': 'Melanoma',
            'nv': 'Melanocytic Nevi',
            'vasc': 'Vascular Lesions'
        }
        
        # Augmentation multipliers for balancing
        self.augmentation_multipliers = {
            'nv': 1,     # Base class
            'mel': 6,    
            'bkl': 6,    
            'bcc': 13,   
            'akiec': 20, 
            'vasc': 47,  
            'df': 58     
        }
        
        # Create directories
        self.create_directories()

    def create_directories(self):
        """Create necessary directories for processed data"""
        print("Creating directories...")
        for split in ['train', 'validation', 'test']:
            for class_name in self.class_names.keys():
                os.makedirs(os.path.join(self.processed_data_path, split, class_name), exist_ok=True)

    def analyze_dataset(self):
        """Analyze dataset distribution"""
        print("\nAnalyzing dataset distribution...")
        metadata = pd.read_csv(os.path.join(self.raw_data_path, 'metadata', 'HAM10000_metadata.csv'))
        
        # Plot original distribution
        plt.figure(figsize=(15, 5))
        distribution = metadata['dx'].value_counts()
        
        plt.subplot(1, 2, 1)
        sns.barplot(x=distribution.index, y=distribution.values)
        plt.title('Original Class Distribution')
        plt.xlabel('Disease Class')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45)
        
        # Print distribution
        print("\nOriginal class distribution:")
        for class_name, count in distribution.items():
            percentage = (count/len(metadata)) * 100
            print(f"{self.class_names[class_name]}: {count} images ({percentage:.2f}%)")
        
        plt.tight_layout()
        plt.show()
        return metadata

    def augment_image(self, image, class_name):
        """Apply augmentation based on class"""
        # Create augmentation generator
        augmentation = ImageDataGenerator(
            rotation_range=360 if class_name in ['vasc', 'df', 'akiec'] else 20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True if class_name in ['vasc', 'df', 'akiec'] else False,
            brightness_range=[0.7, 1.3] if class_name in ['vasc', 'df', 'akiec'] else None,
            fill_mode='nearest'
        )
        
        # Reshape image for augmentation
        image = image.astype('float32') / 255.0
        
        # Generate augmented image
        aug_image = augmentation.random_transform(image)
        
        # Convert back to uint8
        aug_image = (aug_image * 255).astype('uint8')
        
        return aug_image

    def process_data(self):
        """Process and balance the dataset"""
        print("\nStarting data preprocessing...")
        
        # Analyze dataset
        metadata = self.analyze_dataset()
        
        # Split data
        train_df, temp_df = train_test_split(metadata, test_size=0.3, random_state=42, stratify=metadata['dx'])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['dx'])
        
        # Process each split
        splits = {
            'train': train_df,
            'validation': val_df,
            'test': test_df
        }
        
        for split_name, df in splits.items():
            print(f"\nProcessing {split_name} split...")
            self.process_split(df, split_name)
        
        # Verify final distribution
        self.verify_processed_data()

    def process_split(self, df, split_name):
        """Process and augment images for a specific split"""
        total_processed = 0
        
        for class_name in self.class_names.keys():
            print(f"\nProcessing {class_name} class...")
            class_df = df[df['dx'] == class_name]
            target_path = os.path.join(self.processed_data_path, split_name, class_name)
            
            # Process each image with progress bar
            for _, row in tqdm(class_df.iterrows(), total=len(class_df)):
                image_id = row['image_id']
                
                # Check both part_1 and part_2 folders
                possible_paths = [
                    os.path.join(self.raw_data_path, 'images', 'HAM10000_images_part_1', f"{image_id}.jpg"),
                    os.path.join(self.raw_data_path, 'images', 'HAM10000_images_part_2', f"{image_id}.jpg")
                ]
                
                # Process original image
                for img_path in possible_paths:
                    if os.path.exists(img_path):
                        try:
                            # Read and process original image
                            img = cv2.imread(img_path)
                            if img is not None:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                img = cv2.resize(img, self.image_size)
                                
                                # Save original image
                                output_path = os.path.join(target_path, f"{image_id}.jpg")
                                cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                                total_processed += 1
                                
                                # Generate augmented images for training set
                                if split_name == 'train':
                                    num_aug = self.augmentation_multipliers[class_name] - 1
                                    for i in range(num_aug):
                                        try:
                                            aug_img = self.augment_image(img, class_name)
                                            aug_path = os.path.join(target_path, f"{image_id}_aug_{i+1}.jpg")
                                            cv2.imwrite(aug_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                                            total_processed += 1
                                        except Exception as e:
                                            print(f"Error in augmentation for {image_id}: {str(e)}")
                        except Exception as e:
                            print(f"Error processing {image_id}: {str(e)}")
                        break
            
            print(f"Completed processing {class_name}: {total_processed} images")

    def verify_processed_data(self):
        """Verify the processed dataset distribution"""
        print("\nVerifying processed data distribution...")
        
        plt.figure(figsize=(15, 5))
        
        for idx, split in enumerate(['train', 'validation', 'test']):
            class_counts = {}
            total_images = 0
            
            for class_name in self.class_names.keys():
                path = os.path.join(self.processed_data_path, split, class_name)
                count = len(os.listdir(path))
                class_counts[class_name] = count
                total_images += count
            
            plt.subplot(1, 3, idx+1)
            plt.bar(class_counts.keys(), class_counts.values())
            plt.title(f'{split.capitalize()} Set Distribution')
            plt.xlabel('Disease Class')
            plt.ylabel('Number of Images')
            plt.xticks(rotation=45)
            
            print(f"\n{split.capitalize()} set distribution:")
            for class_name, count in class_counts.items():
                percentage = (count/total_images) * 100
                print(f"{self.class_names[class_name]}: {count} images ({percentage:.2f}%)")
        
        plt.tight_layout()
        plt.show()

def main():
    try:
        print("Starting data preprocessing pipeline...")
        preprocessor = DataPreprocessor()
        preprocessor.process_data()
        print("\nPreprocessing completed successfully!")
        
    except Exception as e:
        print(f"\nError during preprocessing: {str(e)}")
        print("\nPlease check:")
        print("1. Dataset availability")
        print("2. Directory structure")
        print("3. Disk space")

if __name__ == "__main__":
    main()