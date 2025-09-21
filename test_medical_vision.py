"""
Test script for the Medical Vision Inference System with structured reporting and PDF generation.
"""

import os
from pathlib import Path
from datetime import datetime
from Models.vision_model.medical_inference import MedicalVisionInference
from Models.vision_model.config import SystemConfig
from Models.processors.report_generator import StructuredReportGenerator
from PIL import Image

def setup_output_directory(base_path="reports/medical_reports"):
    """Setup the output directory structure for medical reports."""
    current_date = datetime.now()
    year_month = current_date.strftime("%Y/%m")
    output_dir = Path(base_path) / year_month
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def generate_output_filename(diagnosis="unspecified"):
    """Generate a unique filename for the report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_diagnosis = "".join(c for c in diagnosis if c.isalnum() or c in (' ', '-', '_')).strip()
    return f"medical_report_{timestamp}_{sanitized_diagnosis}.pdf"

def test_medical_vision_system():
    """Test the medical vision system with structured reporting and PDF generation."""
    
    print("\nInitializing Medical Vision System...")
    
    # Setup paths
    current_dir = Path(__file__).parent
    model_path = current_dir / "Models" / "vision_model" / "llama-3.2-11b-vision-local"
    test_image_path = current_dir / "Models" / "vision_model" / "test_images" / "pneumonia.jpg"
    log_file = current_dir / "medical_vision.log"
    
    print(f"\nInitializing Medical Vision System...")
    print(f"Model path: {model_path}")
    print(f"Test image: {test_image_path}")
    print(f"Log file: {log_file}")
    
    try:
        # Initialize the system
        config = SystemConfig()  # Using default configuration
        inference_system = MedicalVisionInference(
            model_path=model_path,
            config=config,
            log_file=log_file
        )
        
        # Load test image
        print("\nLoading test image...")
        image = Image.open(test_image_path)
        
        # Custom prompt for chest X-ray
        prompt = """
        Please analyze this chest X-ray image in detail. Provide:
        1. Image quality and positioning assessment
        2. Systematic review of anatomical structures
        3. Description of any abnormal findings
        4. Potential diagnoses based on the findings
        5. Recommendations for further evaluation if needed
        Give a detailed, professional medical assessment.
        """
        
        print("\nRunning image analysis...")
        # Analyze the image
        result = inference_system.analyze_image(
            image=image,
            prompt=prompt
        )
        
        # Check results
        if result['success']:
            print("\n✅ Analysis completed successfully!")
            print(f"\nInference time: {result['inference_time']:.2f} seconds")
            print(f"Total processing time: {result['total_time']:.2f} seconds")
            
            print("\n=== Raw Analysis ===")
            print(result['analysis'])
            
            print("\n=== Structured Report ===")
            if 'structured_report' in result:
                report = result['structured_report']
                print("\nCLINICAL FINDINGS:")
                pprint(report['clinical_findings'], indent=2)
                
                print("\nDIAGNOSTIC INTERPRETATION:")
                pprint(report['diagnostic_interpretation'], indent=2)
                
                print("\nPATIENT EXPLANATION:")
                pprint(report['patient_explanation'], indent=2)
            else:
                print("No structured report generated")
                
            if 'report_error' in result:
                print("\n⚠️ Report Generation Error:")
                print(result['report_error'])
        else:
            print("\n❌ Analysis failed!")
            print(f"Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    test_medical_vision_system()