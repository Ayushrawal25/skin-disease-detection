import os
import matplotlib.pyplot as plt

def verify_dataset():
    processed_path = 'data/processed_data'
    splits = ['train', 'validation', 'test']
    
    class_names = {
        'akiec': 'Actinic Keratoses',
        'bcc': 'Basal Cell Carcinoma',
        'bkl': 'Benign Keratosis',
        'df': 'Dermatofibroma',
        'mel': 'Melanoma',
        'nv': 'Melanocytic Nevi',
        'vasc': 'Vascular Lesions'
    }
    
    for split in splits:
        print(f"\n{split.capitalize()} Set:")
        total = 0
        class_counts = {}
        
        for class_name in class_names:
            path = os.path.join(processed_path, split, class_name)
            if os.path.exists(path):
                count = len([f for f in os.listdir(path) if f.endswith('.jpg')])
                class_counts[class_names[class_name]] = count
                total += count
                
        # Print counts and percentages
        for name, count in class_counts.items():
            percentage = (count/total) * 100 if total > 0 else 0
            print(f"{name}: {count} images ({percentage:.2f}%)")
            
        # Plot distribution
        plt.figure(figsize=(10, 5))
        plt.bar(class_counts.keys(), class_counts.values())
        plt.title(f'{split} Set Distribution')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    verify_dataset()