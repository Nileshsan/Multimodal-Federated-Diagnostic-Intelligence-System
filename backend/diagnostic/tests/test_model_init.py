"""
Test model initialization and basic inference.
"""
import os
from pathlib import Path
from django.conf import settings
from vision_model.medical_inference import MedicalVisionInference

def test_model_initialization():
    """Test if the model can be initialized correctly."""
    try:
        # Initialize model
        vision_model = MedicalVisionInference(model_path=settings.VISION_MODEL_PATH)
        print("Model initialized successfully!")
        
        # Load test image
        test_image_path = settings.VISION_MODEL_DIR / 'test_images' / 'pneumonia.jpg'
        if not test_image_path.exists():
            print(f"Test image not found at {test_image_path}")
            return
        
        # Try analysis
        result = vision_model.analyze_medical_image(
            image_path=test_image_path,
            prompt="What abnormalities do you see in this chest X-ray?"
        )
        
        print("\nTest Analysis Result:")
        print("-" * 50)
        print(f"Analysis: {result}")
        
        return True
        
    except Exception as e:
        print(f"Error during model initialization: {str(e)}")
        return False

if __name__ == '__main__':
    success = test_model_initialization()
    print("\nModel test completed:", "SUCCESS" if success else "FAILED")