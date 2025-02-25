import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

class SkinDiseaseModel:
    def __init__(self):
        # Configuration
        self.img_size = (224, 224)
        self.batch_size = 32
        self.epochs = 50
        self.num_classes = 7
        self.initial_learning_rate = 0.001
        
        # Paths
        self.data_path = 'data/processed_data'
        self.model_save_path = 'model/saved_models'
        self.log_dir = 'logs/fit/' + datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Create directories
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Class names
        self.class_names = {
            'akiec': 'Actinic Keratoses',
            'bcc': 'Basal Cell Carcinoma',
            'bkl': 'Benign Keratosis',
            'df': 'Dermatofibroma',
            'mel': 'Melanoma',
            'nv': 'Melanocytic Nevi',
            'vasc': 'Vascular Lesions'
        }
        
        # Initialize model
        self.model = self.build_model()
        
    def build_model(self):
        print("Building model...")
        # Base model - MobileNetV2
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze initial layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        
        # First Dense block
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        # Second Dense block
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        # Output layer
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.initial_learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        model.summary()
        return model
    
    def create_data_generators(self):
        print("Creating data generators...")
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Validation data generator
        valid_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.data_path, 'train'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        validation_generator = valid_datagen.flow_from_directory(
            os.path.join(self.data_path, 'validation'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, validation_generator
    
    def compute_class_weights(self, train_generator):
        """Compute class weights for imbalanced dataset"""
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_generator.classes),
            y=train_generator.classes
        )
        return dict(enumerate(class_weights))
    
    def create_callbacks(self):
        print("Setting up callbacks...")
        callbacks = [
            # Model checkpoint
            ModelCheckpoint(
                os.path.join(self.model_save_path, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            # TensorBoard logging
            TensorBoard(
                log_dir=self.log_dir,
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            )
        ]
        return callbacks
    
    def train(self):
        print("Starting training process...")
        
        # Create data generators
        train_generator, validation_generator = self.create_data_generators()
        
        # Compute class weights
        class_weights = self.compute_class_weights(train_generator)
        
        # Get callbacks
        callbacks = self.create_callbacks()
        
        # Train the model
        history = self.model.fit(
            train_generator,
            epochs=self.epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, validation_generator):
        """Evaluate the model and generate detailed metrics"""
        print("\nEvaluating model...")
        
        # Get predictions
        predictions = self.model.predict(validation_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = validation_generator.classes
        
        # Generate classification report
        print("\nClassification Report:")
        print(classification_report(
            true_classes,
            predicted_classes,
            target_names=list(self.class_names.values())
        ))
        
        # Generate confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names.values(),
                   yticklabels=self.class_names.values())
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_save_path, 'confusion_matrix.png'))
        plt.close()
    
    def plot_training_history(self, history):
        print("Plotting training history...")
        
        # Plot accuracy
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_save_path, 'training_history.png'))
        plt.show()

def main():
    try:
        # Print system information
        print("\nSystem Information:")
        print("TensorFlow version:", tf.__version__)
        print("GPU Available:", bool(tf.config.list_physical_devices('GPU')))
        
        # Initialize and train model
        print("\nInitializing model training...")
        model_trainer = SkinDiseaseModel()
        
        # Train model
        history = model_trainer.train()
        
        # Create data generators for evaluation
        _, validation_generator = model_trainer.create_data_generators()
        
        # Evaluate model
        model_trainer.evaluate_model(validation_generator)
        
        # Plot results
        model_trainer.plot_training_history(history)
        
        print("\nTraining completed successfully!")
        print("Model and plots saved in:", model_trainer.model_save_path)
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("\nPlease check:")
        print("1. Data availability")
        print("2. Memory usage")
        print("3. GPU configuration")

if __name__ == "__main__":
    main()