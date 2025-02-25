import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import time
from datetime import datetime

class SkinDiseaseModel:
    def __init__(self):
        # Enhanced Configuration
        self.img_size = (224, 224)
        self.batch_size = 32
        self.initial_epochs = 50
        self.fine_tune_epochs = 100
        self.num_classes = 7
        self.initial_learning_rate = 0.001
        self.fine_tune_learning_rate = 0.0001
        
        # Paths
        self.data_path = 'data/processed_data'
        self.model_save_path = 'model/saved_models'
        self.log_dir = 'logs/fit/' + datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Create necessary directories
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Class names
        self.class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        
        # Initialize model
        self.model = self.build_model()
        
    def build_model(self):
        print("Building enhanced model...")
        # Base model - MobileNetV2
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Initially freeze all layers
        base_model.trainable = False
        
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        
        # Enhanced classification head
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.initial_learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Store base model reference for fine-tuning
        self.base_model = base_model
        
        return model
    
    def create_data_generators(self):
        print("Creating enhanced data generators...")
        # Enhanced data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=45,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.7, 1.3],
            fill_mode='nearest'
        )
        
        # Validation generator
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
            shuffle=True
        )
        
        return train_generator, validation_generator
    
    def create_callbacks(self, phase):
        print(f"Setting up callbacks for {phase}...")
        callbacks = [
            # Model checkpoint
            ModelCheckpoint(
                os.path.join(self.model_save_path, f'best_model_{phase}.h5'),
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
                log_dir=os.path.join(self.log_dir, phase),
                histogram_freq=1
            )
        ]
        return callbacks
    
    def unfreeze_model(self):
        print("Unfreezing layers for fine-tuning...")
        # Unfreeze all layers
        self.base_model.trainable = True
        
        # Freeze first 100 layers
        for layer in self.base_model.layers[:100]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=self.fine_tune_learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self):
        print("Starting enhanced training process...")
        start_time = time.time()
        
        # Create data generators
        train_generator, validation_generator = self.create_data_generators()
        
        # Calculate class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_generator.classes),
            y=train_generator.classes
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        print("\nPhase 1: Initial training...")
        history1 = self.model.fit(
            train_generator,
            epochs=self.initial_epochs,
            validation_data=validation_generator,
            callbacks=self.create_callbacks('phase1'),
            class_weight=class_weight_dict
        )
        
        print("\nPhase 2: Fine-tuning...")
        self.unfreeze_model()
        history2 = self.model.fit(
            train_generator,
            epochs=self.fine_tune_epochs,
            validation_data=validation_generator,
            callbacks=self.create_callbacks('phase2'),
            class_weight=class_weight_dict
        )
        
        # Calculate training time
        training_time = time.time() - start_time
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = int(training_time % 60)
        
        print(f"\nTraining completed in {hours}h {minutes}m {seconds}s")
        
        # Combine histories
        total_history = {
            'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
            'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
            'loss': history1.history['loss'] + history2.history['loss'],
            'val_loss': history1.history['val_loss'] + history2.history['val_loss']
        }
        
        return total_history
    
    def plot_training_history(self, history):
        print("Plotting training history...")
        plt.figure(figsize=(15, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_save_path, 'training_history.png'))
        plt.show()
        
        # Print best accuracy
        best_val_acc = max(history['val_accuracy'])
        print(f"\nBest validation accuracy: {best_val_acc:.4f}")

def main():
    try:
        # Print system information
        print("\nSystem Information:")
        print("TensorFlow version:", tf.__version__)
        print("GPU Available:", bool(tf.config.list_physical_devices('GPU')))
        
        # Initialize and train model
        print("\nInitializing enhanced model training...")
        model_trainer = SkinDiseaseModel()
        
        # Train model
        history = model_trainer.train()
        
        # Plot results
        model_trainer.plot_training_history(history)
        
        print("\nTraining completed successfully!")
        print("Model and plots saved in:", model_trainer.model_save_path)
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Check available memory")
        print("2. Verify dataset paths")
        print("3. Ensure all dependencies are installed")

if __name__ == "__main__":
    main()