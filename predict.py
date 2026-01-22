"""
Prediction functions for digit recognition
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import preprocess_image
from train_model import DigitRecognitionModel

class DigitPredictor:
    def __init__(self, model_path='../model/resnet50_digit_recognition.h5'):
        """
        Initialize digit predictor
        
        Args:
            model_path: Path to trained model
        """
        self.model_path = model_path
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load trained model"""
        print(f"Loading model from {self.model_path}...")
        
        if os.path.exists(self.model_path):
            # Load using custom handler
            model_handler = DigitRecognitionModel()
            model_handler.load_model(self.model_path)
            self.model = model_handler.model
            print("Model loaded successfully!")
        else:
            print(f"Model not found at {self.model_path}")
            print("Please train the model first using train_model.py")
    
    def predict_from_canvas(self, canvas_image):
        """
        Predict digit from canvas drawing
        
        Args:
            canvas_image: PIL Image or numpy array from canvas
        
        Returns:
            digit: Predicted digit (0-9)
            confidence: Prediction confidence (0-1)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Preprocess canvas image
            processed_image = preprocess_image(
                canvas_image, 
                target_size=(224, 224), 
                model_type='resnet50'
            )
            
            # Add batch dimension
            if len(processed_image.shape) == 3:
                processed_image = np.expand_dims(processed_image, axis=0)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Get predicted digit and confidence
            digit = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            return int(digit), float(confidence)
            
        except Exception as e:
            print(f"Error predicting from canvas: {e}")
            return None, None
    
    def predict_from_image(self, image_path):
        """
        Predict digit from image file
        
        Args:
            image_path: Path to image file
        
        Returns:
            digit: Predicted digit (0-9)
            confidence: Prediction confidence (0-1)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Load image
            if isinstance(image_path, str):
                image = Image.open(image_path)
            else:
                image = image_path
            
            # Preprocess image
            processed_image = preprocess_image(
                image, 
                target_size=(224, 224), 
                model_type='resnet50'
            )
            
            # Add batch dimension
            if len(processed_image.shape) == 3:
                processed_image = np.expand_dims(processed_image, axis=0)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Get predicted digit and confidence
            digit = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            return int(digit), float(confidence)
            
        except Exception as e:
            print(f"Error predicting from image: {e}")
            return None, None
    
    def predict_from_camera_frame(self, frame):
        """
        Predict digit from camera frame
        
        Args:
            frame: Camera frame (numpy array)
        
        Returns:
            digit: Predicted digit (0-9)
            confidence: Prediction confidence (0-1)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Convert BGR to RGB if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Check if it's BGR (OpenCV default)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            # Convert to PIL Image
            image = Image.fromarray(frame_rgb)
            
            # Preprocess image
            processed_image = preprocess_image(
                image, 
                target_size=(224, 224), 
                model_type='resnet50'
            )
            
            # Add batch dimension
            if len(processed_image.shape) == 3:
                processed_image = np.expand_dims(processed_image, axis=0)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Get predicted digit and confidence
            digit = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            return int(digit), float(confidence)
            
        except Exception as e:
            print(f"Error predicting from camera frame: {e}")
            return None, None
    
    def predict_batch(self, images):
        """
        Predict digits for batch of images
        
        Args:
            images: List of images (PIL Images or numpy arrays)
        
        Returns:
            digits: List of predicted digits
            confidences: List of prediction confidences
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Preprocess all images
            processed_images = []
            for image in images:
                processed = preprocess_image(
                    image, 
                    target_size=(224, 224), 
                    model_type='resnet50'
                )
                processed_images.append(processed)
            
            processed_images = np.array(processed_images)
            
            # Make predictions
            predictions = self.model.predict(processed_images, verbose=0)
            
            # Get predicted digits and confidences
            digits = np.argmax(predictions, axis=1).astype(int)
            confidences = np.max(predictions, axis=1).astype(float)
            
            return digits.tolist(), confidences.tolist()
            
        except Exception as e:
            print(f"Error predicting batch: {e}")
            return [], []
    
    def get_top_k_predictions(self, image, k=3):
        """
        Get top-K predictions with confidences
        
        Args:
            image: Input image
            k: Number of top predictions to return
        
        Returns:
            List of tuples (digit, confidence)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Preprocess image
            processed_image = preprocess_image(
                image, 
                target_size=(224, 224), 
                model_type='resnet50'
            )
            
            # Add batch dimension
            if len(processed_image.shape) == 3:
                processed_image = np.expand_dims(processed_image, axis=0)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)[0]
            
            # Get top-K predictions
            top_k_indices = np.argsort(predictions)[::-1][:k]
            top_k_predictions = [
                (int(idx), float(predictions[idx])) 
                for idx in top_k_indices
            ]
            
            return top_k_predictions
            
        except Exception as e:
            print(f"Error getting top-K predictions: {e}")
            return []
    
    def analyze_prediction_confidence(self, predictions, confidence_threshold=0.8):
        """
        Analyze prediction confidence
        
        Args:
            predictions: List of prediction confidences
            confidence_threshold: Threshold for high confidence
        
        Returns:
            Dictionary with analysis results
        """
        predictions = np.array(predictions)
        
        analysis = {
            'mean_confidence': float(np.mean(predictions)),
            'median_confidence': float(np.median(predictions)),
            'std_confidence': float(np.std(predictions)),
            'min_confidence': float(np.min(predictions)),
            'max_confidence': float(np.max(predictions)),
            'high_confidence_count': int(np.sum(predictions >= confidence_threshold)),
            'low_confidence_count': int(np.sum(predictions < confidence_threshold)),
            'high_confidence_percentage': float(np.mean(predictions >= confidence_threshold) * 100)
        }
        
        return analysis

def test_predictor():
    """Test the predictor class"""
    print("Testing DigitPredictor...")
    
    # Initialize predictor
    predictor = DigitPredictor()
    
    if predictor.model is None:
        print("Model not loaded. Skipping tests.")
        return
    
    # Create a test image (simulate a digit)
    test_image = np.zeros((28, 28), dtype=np.uint8)
    
    # Draw a simple '5' shape
    test_image[5:10, 5:20] = 255  # Top horizontal
    test_image[5:20, 5:10] = 255  # Left vertical
    test_image[10:15, 5:20] = 255  # Middle horizontal
    test_image[15:20, 15:20] = 255  # Right vertical
    test_image[20:25, 5:20] = 255  # Bottom horizontal
    
    # Convert to PIL Image
    pil_image = Image.fromarray(test_image)
    
    # Test single prediction
    print("\nTesting single prediction...")
    digit, confidence = predictor.predict_from_canvas(pil_image)
    print(f"  Predicted digit: {digit}")
    print(f"  Confidence: {confidence:.2%}")
    
    # Test top-K predictions
    print("\nTesting top-3 predictions...")
    top_predictions = predictor.get_top_k_predictions(pil_image, k=3)
    for i, (pred_digit, pred_confidence) in enumerate(top_predictions):
        print(f"  Rank {i+1}: Digit {pred_digit} with {pred_confidence:.2%} confidence")
    
    # Test batch prediction
    print("\nTesting batch prediction...")
    test_images = [pil_image, pil_image, pil_image]  # Same image 3 times
    digits, confidences = predictor.predict_batch(test_images)
    print(f"  Batch predictions: {digits}")
    print(f"  Confidences: {[f'{c:.2%}' for c in confidences]}")
    
    # Analyze confidence
    print("\nAnalyzing confidence...")
    analysis = predictor.analyze_prediction_confidence(confidences)
    for key, value in analysis.items():
        if 'percentage' in key:
            print(f"  {key}: {value:.1f}%")
        elif 'count' in key:
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.4f}")
    
    print("\nPredictor tests completed!")

if __name__ == "__main__":
    test_predictor()
