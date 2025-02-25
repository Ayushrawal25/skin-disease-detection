import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_dataset():
    # Read metadata
    metadata = pd.read_csv('data/raw_data/HAM10000_metadata.csv')
    
    # Class names mapping
    class_names = {
        'akiec': 'Actinic Keratoses',
        'bcc': 'Basal Cell Carcinoma',
        'bkl': 'Benign Keratosis',
        'df': 'Dermatofibroma',
        'mel': 'Melanoma',
        'nv': 'Melanocytic Nevi',
        'vasc': 'Vascular Lesions'
    }
    
    # Print basic info
    print("\nDataset Overview:")
    print(f"Total images: {len(metadata)}")
    
    # Show class distribution
    print("\nClass Distribution:")
    distribution = metadata['dx'].value_counts()
    for class_name, count in distribution.items():
        percentage = (count/len(metadata)) * 100
        print(f"{class_names[class_name]}: {count} images ({percentage:.2f}%)")
    
    # Plot distribution
    plt.figure(figsize=(12, 6))
    sns.barplot(x=distribution.index, y=distribution.values)
    plt.title('Distribution of Classes in Dataset')
    plt.xlabel('Disease Class')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_dataset()