#!/usr/bin/env python3
"""Create a test model for the API"""

import tensorflow as tf
import numpy as np
import os

def create_test_model():
    """Create a test model matching the expected architecture"""
    print("🔧 Creating test model...")
    
    # Create model with expected architecture
    inputs = tf.keras.Input(shape=(360, 360, 1), name='input')
    
    # Simple CNN architecture
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    
    # Age output (regression)
    age_output = tf.keras.layers.Dense(1, activation='linear', name='age_output')(x)
    
    # Gender output (binary classification) 
    gender_output = tf.keras.layers.Dense(1, activation='sigmoid', name='gender_output')(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=[age_output, gender_output])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss={'age_output': 'mse', 'gender_output': 'binary_crossentropy'},
        metrics={'age_output': 'mae', 'gender_output': 'accuracy'}
    )
    
    # Initialize with random weights
    dummy_input = np.random.random((1, 360, 360, 1))
    _ = model.predict(dummy_input, verbose=0)
    
    # Save model
    model_path = 'best_age_gender_model_children_tuned.h5'
    model.save(model_path)
    
    print(f"✅ Test model saved as: {model_path}")
    print(f"📊 Model summary:")
    print(f"   - Input shape: {model.input_shape}")
    print(f"   - Output shapes: {[output.shape for output in model.outputs]}")
    print(f"   - Total parameters: {model.count_params():,}")
    
    # Test the model
    print("\n🧪 Testing model...")
    test_input = np.random.random((1, 360, 360, 1))
    predictions = model.predict(test_input, verbose=0)
    age_pred = predictions[0][0][0]
    gender_pred = predictions[1][0][0]
    
    print(f"   - Sample age prediction: {age_pred:.1f}")
    print(f"   - Sample gender prediction: {gender_pred:.3f}")
    
    print("\n⚠️  NOTE: This is a test model with random weights!")
    print("   Predictions will be random until you train with real data.")
    
    return model_path

if __name__ == "__main__":
    create_test_model()