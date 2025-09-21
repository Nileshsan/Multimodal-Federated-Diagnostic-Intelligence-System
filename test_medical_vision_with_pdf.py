"""
Test script for medical vision system with PDF report generation.
"""

import os
from pathlib import Path
from datetime import datetime
from Models.Medical_Diagnostic_Vision_System import DiagnosticVisionSystem
from Models.processors.pdf_generator import save_report_as_pdf

def main():
    print("\nInitializing Medical Vision System...")
    
    # Initialize the vision system
    vision_system = DiagnosticVisionSystem()
    
    # Set up paths
    model_path = Path("Models/vision_model/llama-3.2-11b-vision-local")
    test_image = Path("Models/vision_model/test_images/pneumonia.jpg")
    log_file = Path("medical_vision.log")
    
    print(f"Model path: {model_path.absolute()}")
    print(f"Test image: {test_image.absolute()}")
    print(f"Log file: {log_file.absolute()}")
    
    try:
        # Run the analysis
        start_time = datetime.now()
        report = vision_system.analyze_image(str(test_image.absolute()))
        end_time = datetime.now()
        
        # Calculate processing times
        inference_time = (end_time - start_time).total_seconds()
        
        print("\n✅ Analysis completed successfully!")
        print(f"\nInference time: {inference_time:.2f} seconds")
        
        # Generate and save PDF report
        pdf_path = save_report_as_pdf(report)
        
        print("\nProcess completed successfully!")
        print("="*50)
        print(f"PDF Report saved at: {pdf_path}")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()