# SPDX-License-Identifier: MIT
# Copyright © 2025 github.com/dtiberio

# Debug script to test model loading and basic prediction
import os
import joblib
import pandas as pd
import numpy as np

def debug_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=== MODEL DEBUG SCRIPT ===")
    print(f"Script directory: {script_dir}")
    
    # Check if model files exist
    model_files = [
        'heart_disease_model.pkl',
        'feature_names.pkl', 
        'model_metrics.pkl'
    ]
    
    print("\n1. Checking model files:")
    for file in model_files:
        file_path = os.path.join(script_dir, file)
        exists = os.path.exists(file_path)
        print(f"   {file}: {'✓' if exists else '✗'}")
        if not exists:
            print(f"      Missing: {file_path}")
    
    try:
        # Load model components
        print("\n2. Loading model components:")
        model = joblib.load(os.path.join(script_dir, 'heart_disease_model.pkl'))
        feature_names = joblib.load(os.path.join(script_dir, 'feature_names.pkl'))
        model_metrics = joblib.load(os.path.join(script_dir, 'model_metrics.pkl'))
        
        print(f"   Model type: {type(model)}")
        print(f"   Feature names: {feature_names}")
        print(f"   Number of features: {len(feature_names)}")
        print(f"   Model metrics keys: {list(model_metrics.keys()) if isinstance(model_metrics, dict) else 'Not a dict'}")
        
        # Test with sample data
        print("\n3. Testing with sample data:")
        sample_data = {
            'age': 50,
            'sex': 1,
            'cp': 0,
            'trestbps': 120,
            'chol': 200,
            'fbs': 0,
            'restecg': 0,
            'thalach': 150,
            'exang': 0,
            'oldpeak': 1.0,
            'slope': 0,
            'ca': 0,
            'thal': 1
        }
        
        print(f"   Sample data: {sample_data}")
        print(f"   Sample data keys: {list(sample_data.keys())}")
        print(f"   Expected features: {feature_names}")
        
        # Check for missing features
        missing = set(feature_names) - set(sample_data.keys())
        extra = set(sample_data.keys()) - set(feature_names)
        
        if missing:
            print(f"   ❌ Missing features: {missing}")
        if extra:
            print(f"   ⚠️ Extra features: {extra}")
        
        # Create DataFrame
        input_df = pd.DataFrame([sample_data])[feature_names]
        print(f"   Input DataFrame shape: {input_df.shape}")
        print(f"   Input DataFrame dtypes:\n{input_df.dtypes}")
        
        # Make prediction
        print("\n4. Making prediction:")
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)
        
        print(f"   Prediction: {prediction}")
        print(f"   Probability shape: {probability.shape}")
        print(f"   Probability: {probability}")
        
        print("\n✅ Model test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during model testing:")
        print(f"   Error: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        
        import traceback
        print(f"\n   Full traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    debug_model()