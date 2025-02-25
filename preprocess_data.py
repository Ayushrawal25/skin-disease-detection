import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil

class DataPreprocessor:
    def __init__(self):
        self.raw_data_path = 'data/raw_data'
        self.processed_data_path = 'data/processed_data'
        self.image_size = (224, 224)
        
        # Class mapping
        self.class_names = {
            'akiec': 'Actinic Keratoses',
            'bcc': 'Basal Cell Carcinoma',
            'bkl': 'Benign Keratosis',
            'df': 'Dermatofibroma',
            'mel': 'Melanoma',
            'nv': 'Melanocytic Nevi',
            'vasc': 'Vascular Lesions'
        }
        
        # Clean and create directories
        self.clean_directories()
        self.create_directories()

    def clean_directories(self):
        """Clean existing processed data"""
        if os.path.exists(self.processed_data_path):
            print("Cleaning existing processed data...")
            shutil.rmtree(self.processed_data_path)

    def create_directories(self):
        """Create necessary directories"""
        print("Creating directories...")
        for split in ['train', 'validation', 'test']:
            for class_name in self.class_names.keys():
                path = os.path.join(self.processed_data_path, split, class_name)
                os.makedirs(path, exist_ok=True)

    def find_image(self, image_id):
        """Find image in HAM10000 parts"""
        possible_paths = [
            os.path.join(self.raw_data_path, 'HAM10000_images_part_1', f'{image_id}.jpg'),
            os.path.join(self.raw_data_path, 'HAM10000_images_part_2', f'{image_id}.jpg')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None

    def augment_image(self, image):
        """Apply augmentation to image"""
        augmentor = ImageDataGenerator(
            rotation_range=180,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.7, 1.3],
            fill_mode='nearest'
        )
        
        # Prepare image for augmentation
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, 0)
        
        # Generate augmented image
        aug_image = augmentor.random_transform(image[0])
        aug_image = (aug_image * 255).astype('uint8')
        
        return aug_image

    def process_data(self):
        """Main processing function"""
        print("Starting data preprocessing...")
        
        # Read metadata
        metadata_path = os.path.join(self.raw_data_path, 'HAM10000_metadata.csv')
        metadata = pd.read_csv(metadata_path)
        
        # Print initial distribution
        print("\nOriginal data distribution:")
        for class_name, count in metadata['dx'].value_counts().items():
            percentage = (count/len(metadata)) * 100
            print(f"{self.class_names[class_name]}: {count} images ({percentage:.2f}%)")
        
        # Split data
        train_df, temp_df = train_test_split(
            metadata, 
            test_size=0.3,
            stratify=metadata['dx'],
            random_state=42
        )
        
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            stratify=temp_df['dx'],
            random_state=42
        )
        
        # Process each split
        print("\nProcessing training set...")
        self.process_split(train_df, 'train', augment=True)
        
        print("\nProcessing validation set...")
        self.process_split(val_df, 'validation', augment=False)
        
        print("\nProcessing test set...")
        self.process_split(test_df, 'test', augment=False)
        
        # Verify results
        self.verify_processed_data()

    def process_split(self, df, split_name, augment=False):
        """Process and augment images for a specific split"""
        target_samples = 4700 if augment else None  # Target for balanced training set
        
        for class_name in self.class_names.keys():
            print(f"\nProcessing {class_name}...")
            class_df = df[df['dx'] == class_name]
            target_dir = os.path.join(self.processed_data_path, split_name, class_name)
            
            # Process each image
            for _, row in tqdm(class_df.iterrows(), desc="Processing"):
                image_id = row['image_id']
                img_path = self.find_image(image_id)
                
                if img_path is not None:
                    # Read and preprocess image
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, self.image_size)
                        
                        # Save original image
                        save_path = os.path.join(target_dir, f"{image_id}.jpg")
                        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        
                        # Generate augmented images for training
                        if augment and target_samples:
                            current_count = len(class_df)
                            if current_count < target_samples:
                                num_aug = (target_samples // current_count)
                                for i in range(num_aug):
                                    aug_img = self.augment_image(img)
                                    aug_path = os.path.join(target_dir, f"{image_id}_aug_{i}.jpg")
                                    cv2.imwrite(aug_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

    def verify_processed_data(self):
        """Verify and visualize processed data"""
        print("\nVerifying processed data distribution...")
        
        plt.figure(figsize=(15, 5))
        
        for idx, split in enumerate(['train', 'validation', 'test']):
            class_counts = {}
            total_images = 0
            
            # Count images in each class
            for class_name in self.class_names.keys():
                path = os.path.join(self.processed_data_path, split, class_name)
                count = len([f for f in os.listdir(path) if f.endswith('.jpg')])
                class_counts[class_name] = count
                total_images += count
            
            # Print statistics
            print(f"\n{split.capitalize()} set distribution:")
            for class_name, count in class_counts.items():
                percentage = (count/total_images) * 100 if total_images > 0 else 0
                print(f"{self.class_names[class_name]}: {count} images ({percentage:.2f}%)")
            
            # Plot distribution
            plt.subplot(1, 3, idx+1)
            plt.bar(class_counts.keys(), class_counts.values())
            plt.title(f'{split.capitalize()} Set Distribution')
            plt.xlabel('Disease Class')
            plt.ylabel('Number of Images')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()

def main():
    try:
        preprocessor = DataPreprocessor()
        preprocessor.process_data()
        print("\nPreprocessing completed successfully!")
        
    except Exception as e:
        print(f"\nError during preprocessing: {str(e)}")
        print("\nPlease check:")
        print("1. Data availability")
        print("2. Directory structure")
        print("3. Disk space")

if __name__ == "__main__":
    main()