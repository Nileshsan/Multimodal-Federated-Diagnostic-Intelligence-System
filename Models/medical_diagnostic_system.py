"""
Medical Diagnostic System
Handles multi-modal medical data analysis and diagnostic report generation.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from PIL import Image
import torch
import numpy as np

from Models.processors.image_processor import ImageProcessor
from Models.processors.text_processor import TextProcessor

# Add project root to path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.append(project_root)

from Models.vision_model.medical_inference import MedicalVisionInference
from Models.vision_model.config import SystemConfig
from Models.vision_model.logger import MedicalVisionLogger
from Models.vision_model.errors import (
    ModelLoadError, ValidationError, InferenceError
)

class InputType:
    """Enumeration of supported input types."""
    IMAGE = "image"
    PDF = "pdf"
    TEXT = "text"
    UNKNOWN = "unknown"

class ImageType:
    """Enumeration of supported medical image types."""
    XRAY = "xray"
    CT = "ct"
    MRI = "mri"
    ULTRASOUND = "ultrasound"
    UNKNOWN = "unknown"

class MedicalDiagnosticSystem:
    """Core system for medical diagnostic analysis."""
    
    def __init__(
        self,
        model_path: Union[str, Path],
        config: Optional[SystemConfig] = None,
        log_file: Optional[Path] = None
    ):
        """Initialize the medical diagnostic system.
        
        Args:
            model_path: Path to the base vision model
            config: Optional system configuration
            log_file: Optional log file path
        """
        self.logger = MedicalVisionLogger("MedicalDiagnostic", log_file)
        self.config = config or SystemConfig()
        
        # Initialize vision model
        self.vision_model = MedicalVisionInference(
            model_path=model_path,
            config=self.config,
            log_file=log_file
        )
        
        # Initialize processors
        self.image_processor = ImageProcessor(
            device=self.vision_model.device if hasattr(self.vision_model, 'device') else None
        )
        self.text_processor = TextProcessor()
        
        # Cache for analysis results
        self.cache = {}
        
        self.logger.info("Medical Diagnostic System initialized")
    
    def detect_input_type(self, file_path: Union[str, Path]) -> str:
        """Detect the type of input file.
        
        Args:
            file_path: Path to input file
            
        Returns:
            str: Input type (image, pdf, text)
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        # Image types
        if extension in ['.png', '.jpg', '.jpeg', '.dicom', '.dcm']:
            return InputType.IMAGE
        # PDF type
        elif extension == '.pdf':
            return InputType.PDF
        # Text type
        elif extension in ['.txt', '.doc', '.docx']:
            return InputType.TEXT
        else:
            return InputType.UNKNOWN
    
    def classify_image_type(self, image_path: Union[str, Path]) -> str:
        """Classify the type of medical image.
        
        Args:
            image_path: Path to medical image
            
        Returns:
            str: Image type classification
        """
        try:
            # Use vision model to get initial analysis
            analysis = self.vision_model.analyze_medical_image(image_path)
            
            # Extract image type from analysis
            if analysis and analysis.get('success', False):
                text = analysis['analysis'].lower()
                
                # Simple keyword-based classification
                if any(word in text for word in ['x-ray', 'xray', 'radiograph']):
                    return ImageType.XRAY
                elif any(word in text for word in ['ct', 'computed tomography']):
                    return ImageType.CT
                elif any(word in text for word in ['mri', 'magnetic resonance']):
                    return ImageType.MRI
                elif 'ultrasound' in text:
                    return ImageType.ULTRASOUND
            
            return ImageType.UNKNOWN
            
        except Exception as e:
            self.logger.error(f"Error classifying image type: {str(e)}")
            return ImageType.UNKNOWN
    
    def extract_text_from_pdf(self, pdf_path: Union[str, Path]) -> str:
        """Extract text content from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            str: Extracted text content
        """
        try:
            text = []
            pdf_doc = fitz.open(pdf_path)
            
            for page in pdf_doc:
                text.append(page.get_text())
            
            return "\n".join(text)
            
        except Exception as e:
            self.logger.error(f"Error extracting PDF text: {str(e)}")
            return ""
    
    def analyze_input(
        self,
        input_path: Union[str, Path],
        symptoms: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze medical input and generate diagnostic information.
        
        Args:
            input_path: Path to input file (image/PDF/text)
            symptoms: Optional symptom description
            
        Returns:
            dict: Analysis results including:
                - input_type: Type of input analyzed
                - analysis: Main analysis results
                - image_type: Type of medical image (if applicable)
                - extracted_text: Extracted text (if PDF)
                - symptom_analysis: Analysis of provided symptoms
        """
        try:
            results = {
                'success': False,
                'input_type': self.detect_input_type(input_path),
                'timestamp': time.time()
            }
            
            # Process based on input type
            if results['input_type'] == InputType.IMAGE:
                # Process image using specialized processor
                image_data = self.image_processor.load_image(input_path)
                results['image_type'] = image_data['format']
                results['image_info'] = self.image_processor.get_image_info(image_data)
                
                # Analyze with vision model
                analysis = self.vision_model.analyze_medical_image(input_path)
                if analysis and analysis.get('success', False):
                    results['analysis'] = analysis['analysis']
                    results['success'] = True
                    
                    # Cache successful analysis
                    self.cache[str(input_path)] = {
                        'timestamp': results['timestamp'],
                        'analysis': analysis['analysis'],
                        'image_type': results['image_type']
                    }
                
            elif results['input_type'] == InputType.TEXT:
                # Read and analyze text
                with open(input_path, 'r') as f:
                    text = f.read()
                results['text_content'] = text
                
                # Analyze text content
                text_analysis = self.text_processor.analyze_medical_text(text)
                results['text_analysis'] = text_analysis
                results['success'] = True
            
            # Add symptom analysis if provided
            if symptoms:
                symptom_analysis = self.text_processor.analyze_medical_text(symptoms)
                results['symptom_analysis'] = symptom_analysis
                
                # Update success flag if we have any valid results
                if symptom_analysis['symptoms']['symptoms']:
                    results['success'] = True
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing input: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def generate_report(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive medical report from analysis results.
        
        Args:
            analysis_results: Results from analyze_input
            
        Returns:
            dict: Formatted report including:
                - summary: Brief summary of findings
                - details: Detailed analysis
                - recommendations: Suggested next steps
        """
        try:
            report = {
                'timestamp': analysis_results.get('timestamp', ''),
                'input_type': analysis_results.get('input_type', ''),
                'summary': '',
                'details': {},
                'recommendations': []
            }
            
            # Generate summary based on input type
            if analysis_results['input_type'] == InputType.IMAGE:
                report['details']['image_type'] = analysis_results.get('image_type', '')
                report['details']['image_analysis'] = analysis_results.get('analysis', '')
                
                # Extract key findings from analysis
                if 'analysis' in analysis_results:
                    analysis_text = analysis_results['analysis']
                    # TODO: Implement better summary extraction
                    report['summary'] = analysis_text[:200] + "..."
                    
                    # Basic recommendations based on image type
                    if analysis_results['image_type'] == ImageType.XRAY:
                        report['recommendations'].append(
                            "Consider follow-up chest X-ray in 6-8 weeks if symptoms persist"
                        )
            
            elif analysis_results['input_type'] == InputType.PDF:
                report['details']['extracted_text'] = analysis_results.get('extracted_text', '')
                # TODO: Implement PDF report analysis
                
            # Add symptom analysis if available
            if 'symptom_analysis' in analysis_results:
                report['details']['symptoms'] = analysis_results['symptom_analysis']
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return {
                'error': f"Failed to generate report: {str(e)}",
                'timestamp': analysis_results.get('timestamp', '')
            }

def main():
    """Test the medical diagnostic system."""
    try:
        # Initialize system
        model_path = Path("Models/vision_model/llama-3.2-11b-vision-local")
        system = MedicalDiagnosticSystem(
            model_path=model_path,
            log_file=Path("medical_diagnostic.log")
        )
        
        # Test with sample image
        test_image = Path("Models/vision_model/test_images/pneumonia.jpg")
        if test_image.exists():
            # Test analysis with symptoms
            results = system.analyze_input(
                test_image,
                symptoms="Patient reports chest pain and difficulty breathing for 3 days"
            )
            
            if results['success']:
                # Generate report
                report = system.generate_report(results)
                
                print("\n=== Medical Analysis Report ===")
                print(f"Input Type: {results['input_type']}")
                print(f"Image Type: {results.get('image_type', 'N/A')}")
                print("\nAnalysis:")
                print(results.get('analysis', 'No analysis available'))
                
                if 'symptom_analysis' in results:
                    print("\nSymptom Analysis:")
                    print(results['symptom_analysis'].get('preliminary_assessment', ''))
                
                print("\nRecommendations:")
                for rec in report.get('recommendations', []):
                    print(f"- {rec}")
            else:
                print("Analysis failed:", results.get('error', 'Unknown error'))
        else:
            print(f"Test image not found: {test_image}")
            
    except Exception as e:
        print(f"Test failed: {str(e)}")

if __name__ == "__main__":
    main()