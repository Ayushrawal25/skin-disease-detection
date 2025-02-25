import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

class SkinDiseasePredictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.img_size = (224, 224)
        self.class_names = {
            'akiec': 'Actinic Keratoses',
            'bcc': 'Basal Cell Carcinoma',
            'bkl': 'Benign Keratosis',
            'df': 'Dermatofibroma',
            'mel': 'Melanoma',
            'nv': 'Melanocytic Nevi',
            'vasc': 'Vascular Lesions'
        }

    def preprocess_image(self, image_path):
        # Read and preprocess image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize while maintaining aspect ratio
        height, width = img.shape[:2]
        aspect_ratio = width / height
        if aspect_ratio > 1:
            new_width = self.img_size[0]
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = self.img_size[1]
            new_width = int(new_height * aspect_ratio)
            
        img_resized = cv2.resize(img, (new_width, new_height))
        
        # Create blank image
        img_padded = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
        
        # Center the image
        y_offset = (self.img_size[0] - new_height) // 2
        x_offset = (self.img_size[1] - new_width) // 2
        img_padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = img_resized
        
        # Normalize
        img_processed = img_padded.astype('float32') / 255.0
        
        return img_processed, img

    def predict(self, image_path):
        # Preprocess image
        img_processed, img_original = self.preprocess_image(image_path)
        
        # Add batch dimension
        img_batch = np.expand_dims(img_processed, axis=0)
        
        # Get predictions
        predictions = self.model.predict(img_batch)
        
        # Get top 3 predictions
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            (self.class_names[list(self.class_names.keys())[idx]], 
             predictions[0][idx] * 100) 
            for idx in top_3_idx
        ]
        
        # Create visualization
        plt.figure(figsize=(15, 8))
        
        # Original Image
        plt.subplot(1, 2, 1)
        plt.imshow(img_original)
        plt.title('Original Image', pad=20)
        plt.axis('off')
        
        # Predictions Bar Chart
        plt.subplot(1, 2, 2)
        
        # Get all predictions sorted
        probs_dict = {self.class_names[k]: float(v) for k, v in zip(self.class_names.keys(), predictions[0])}
        sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
        
        names = [x[0] for x in sorted_probs]
        values = [x[1] for x in sorted_probs]
        
        colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(names))]
        bars = plt.barh(range(len(names)), [val * 100 for val in values], color=colors)
        plt.yticks(range(len(names)), names)
        plt.xlabel('Confidence (%)')
        
        # Add percentage labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}%',
                    ha='left', va='center', fontweight='bold')
        
        plt.title('Prediction Confidence Levels', pad=20)
        
        # Add main title with top prediction
        plt.suptitle(f"Predicted: {top_3_predictions[0][0]}\nConfidence: {top_3_predictions[0][1]:.2f}%",
                    fontsize=16, y=1.05)
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed results
        print("\nTop 3 Predictions:")
        for disease, confidence in top_3_predictions:
            print(f"{disease}: {confidence:.2f}%")
        
        # Confidence assessment
        if top_3_predictions[0][1] < 70:
            print("\nNote: Low confidence prediction")
            print("Consider:")
            print("- Using a clearer image")
            print("- Different lighting conditions")
            print("- Multiple angles")
            print("- Consulting a healthcare professional")

def main():
    # Use the latest trained model
    model_path = 'model/saved_models/best_model_phase2.h5'
    
    # Initialize predictor
    predictor = SkinDiseasePredictor(model_path)
    
    # Test image path - update this for each test
    image_path = r"D:\major project\Test Images\mel2.jpg"  # Use raw string for Windows path
    
    # Make prediction
    predictor.predict(image_path)

if __name__ == "__main__":
    main()