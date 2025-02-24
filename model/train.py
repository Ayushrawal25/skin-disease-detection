# In the SkinDiseaseModel class, update these parameters:
def __init__(self):
    # Configuration
    self.img_size = (224, 224)
    self.batch_size = 8  # Reduced batch size for CPU
    self.epochs = 30
    self.num_classes = 7
    self.initial_learning_rate = 0.001
    
    # Paths
    self.data_path = 'data/processed_data'
    self.model_save_path = 'model/saved_models'
    os.makedirs(self.model_save_path, exist_ok=True)
    
    # Initialize model
    self.model = self.build_model()

def train(self):
    print("Starting training process...")
    # Create data generators
    train_generator, validation_generator = self.create_data_generators()
    
    # Get callbacks
    callbacks = self.create_callbacks()
    
    # Calculate steps per epoch
    steps_per_epoch = train_generator.samples // self.batch_size
    validation_steps = validation_generator.samples // self.batch_size
    
    # Train the model
    history = self.model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=self.epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        workers=1,  # Reduced for CPU
        use_multiprocessing=False  # Disabled for CPU
    )
    
    return history

# Update the main function
def main():
    try:
        print("\nInitializing model training...")
        print("Training will proceed on CPU")
        model_trainer = SkinDiseaseModel()
        
        # Train model
        history = model_trainer.train()
        
        # Plot results
        model_trainer.plot_training_history(history)
        
        print("\nTraining completed successfully!")
        print("Model saved in:", model_trainer.model_save_path)
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("\nPlease check:")
        print("1. Available memory")
        print("2. Dataset availability")