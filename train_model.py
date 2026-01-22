"""
Train ResNet-50 model for handwritten digit recognition on MNIST dataset
"""

import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import preprocess_images, load_mnist_data, plot_training_history

class DigitRecognitionModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=10):
        """
        Initialize the ResNet-50 based digit recognition model
        
        Args:
            input_shape: Input image dimensions (height, width, channels)
            num_classes: Number of digit classes (0-9)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_model(self):
        """
        Build ResNet-50 model with custom classification head
        """
        # Load pre-trained ResNet-50 without top layers
        base_model = applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Create custom classification head
        inputs = tf.keras.Input(shape=self.input_shape)
        
        # Preprocessing (ResNet-50 expects specific preprocessing)
        x = applications.resnet.preprocess_input(inputs)
        
        # Base model
        x = base_model(x, training=False)
        
        # Custom layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        self.model = models.Model(inputs, outputs)
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, 
              batch_size=32, epochs=50, use_augmentation=True):
        """
        Train the model
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            batch_size: Batch size for training
            epochs: Number of training epochs
            use_augmentation: Whether to use data augmentation
        """
        # Data augmentation
        if use_augmentation:
            data_augmentation = tf.keras.Sequential([
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
                layers.RandomTranslation(0.1, 0.1),
                layers.RandomContrast(0.1),
            ])
            
            # Create augmented dataset
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            train_dataset = train_dataset.map(
                lambda x, y: (data_augmentation(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            train_dataset = train_dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        else:
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            train_dataset = train_dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath='../model/checkpoints/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train the model
        print("Starting model training...")
        self.history = self.model.fit(
            train_dataset,
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Fine-tuning: Unfreeze some layers
        print("Starting fine-tuning...")
        base_model = self.model.layers[2]  # ResNet-50 layer
        base_model.trainable = True
        
        # Freeze first 100 layers, unfreeze the rest
        for layer in base_model.layers[:100]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Fine-tune for a few epochs
        fine_tune_epochs = 10
        total_epochs = epochs + fine_tune_epochs
        
        history_fine = self.model.fit(
            train_dataset,
            validation_data=(X_val, y_val),
            initial_epoch=self.history.epoch[-1],
            epochs=total_epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Combine histories
        for key in self.history.history.keys():
            self.history.history[key] += history_fine.history[key]
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set
        
        Args:
            X_test: Test images
            y_test: Test labels
        
        Returns:
            test_loss: Test loss
            test_accuracy: Test accuracy
        """
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Get predictions
        y_pred = np.argmax(self.model.predict(X_test), axis=1)
        
        # Generate classification report
        from sklearn.metrics import classification_report, confusion_matrix
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Accuracy (%): {test_accuracy * 100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        return test_loss, test_accuracy
    
    def save_model(self, filepath='../model/resnet50_digit_recognition.h5'):
        """
        Save trained model
        
        Args:
            filepath: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        
        # Also save weights only
        weights_path = filepath.replace('.h5', '_weights.h5')
        self.model.save_weights(weights_path)
        print(f"Model weights saved to {weights_path}")
    
    def load_model(self, filepath='../model/resnet50_digit_recognition.h5'):
        """
        Load trained model
        
        Args:
            filepath: Path to load the model from
        """
        if os.path.exists(filepath):
            self.model = tf.keras.models.load_model(filepath)
            print(f"Model loaded from {filepath}")
        else:
            print(f"Model file not found: {filepath}")
            self.build_model()
    
    def predict(self, image):
        """
        Predict digit from single image
        
        Args:
            image: Preprocessed image tensor
        
        Returns:
            digit: Predicted digit (0-9)
            confidence: Prediction confidence
        """
        # Add batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Make prediction
        predictions = self.model.predict(image, verbose=0)
        
        # Get predicted digit and confidence
        digit = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        return digit, confidence
    
    def predict_batch(self, images):
        """
        Predict digits for a batch of images
        
        Args:
            images: Batch of preprocessed images
        
        Returns:
            digits: List of predicted digits
            confidences: List of prediction confidences
        """
        predictions = self.model.predict(images, verbose=0)
        
        digits = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        
        return digits, confidences

def main():
    """
    Main training function
    """
    print("="*60)
    print("HANDWRITTEN DIGIT RECOGNITION MODEL TRAINING")
    print("Using ResNet-50 Architecture")
    print("="*60)
    
    # Load and preprocess MNIST data
    print("\n1. Loading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = load_mnist_data()
    
    # Split training data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Preprocess images for ResNet-50
    print("\n2. Preprocessing images...")
    X_train = preprocess_images(X_train, target_size=(224, 224), model_type='resnet50')
    X_val = preprocess_images(X_val, target_size=(224, 224), model_type='resnet50')
    X_test = preprocess_images(X_test, target_size=(224, 224), model_type='resnet50')
    
    # Build and train model
    print("\n3. Building ResNet-50 model...")
    model = DigitRecognitionModel()
    model.build_model()
    
    # Display model summary
    print("\nModel Architecture:")
    model.model.summary()
    
    # Train model
    print("\n4. Training model...")
    history = model.train(
        X_train, y_train, 
        X_val, y_val,
        batch_size=32,
        epochs=30,
        use_augmentation=True
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    print("\n5. Evaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    
    # Save model
    print("\n6. Saving trained model...")
    model.save_model()
    
    # Test prediction on sample images
    print("\n7. Testing model predictions...")
    test_sample = X_test[:5]
    test_labels = y_test[:5]
    
    digits, confidences = model.predict_batch(test_sample)
    
    print("\nSample Predictions:")
    for i in range(len(test_sample)):
        print(f"  Image {i+1}: True={test_labels[i]}, Predicted={digits[i]}, "
              f"Confidence={confidences[i]:.2%}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print(f"Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print("="*60)

if __name__ == "__main__":
    main()
