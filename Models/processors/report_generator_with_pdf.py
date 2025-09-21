"""
Structured Medical Report Generator
Handles the generation of detailed medical reports with both technical and patient-friendly information.

This module implements a comprehensive medical report generation system with:
- Structured clinical findings analysis
- Evidence-based diagnostic interpretation
- Medical knowledge base integration
- Patient-friendly explanations
- Proper medical validation and safety checks
"""

from typing import Dict, List, Optional, Union, Any
import json
from pathlib import Path
import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
import logging
from concurrent.futures import ThreadPoolExecutor
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiagnosticConfidence(Enum):
    """Enumeration for diagnostic confidence levels."""
    HIGH = auto()      # >90% confidence
    MODERATE = auto()  # 70-90% confidence
    LOW = auto()       # 50-70% confidence
    UNCERTAIN = auto() # <50% confidence

class MedicalReportError(Exception):
    """Base exception class for medical report generation errors."""
    pass

class ValidationError(MedicalReportError):
    """Raised when report validation fails."""
    pass

class KnowledgeBaseError(MedicalReportError):
    """Raised when knowledge base access fails."""
    pass

@dataclass
class MedicalReport:
    """
    Structured medical report format with comprehensive validation.
    
    Attributes:
        clinical_findings: Detailed clinical observations and measurements
        diagnostic_interpretation: Medical interpretation with confidence levels
        technical_details: Technical aspects of the imaging and analysis
        patient_explanation: Patient-friendly version of findings
        additional_notes: Supplementary information and recommendations
        metadata: Report tracking and versioning information
        validation_status: Report validation state and safety checks
    """
    clinical_findings: Dict
    diagnostic_interpretation: Dict
    technical_details: Dict
    patient_explanation: Dict
    additional_notes: Dict
    metadata: Dict
    validation_status: Dict = field(default_factory=dict)

class StructuredReportGenerator:
    """Generates structured medical reports from model outputs with safety validation."""

    def __init__(
        self,
        medical_knowledge_base: Optional[Path] = None,
        safety_checks: bool = True,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize the report generator with safety features.

        Args:
            medical_knowledge_base: Path to medical knowledge base JSON
            safety_checks: Enable strict medical safety validation
            confidence_threshold: Minimum confidence for diagnostic claims
        """
        self.safety_checks = safety_checks
        self.confidence_threshold = confidence_threshold
        self.knowledge_base = self._load_knowledge_base(medical_knowledge_base)
        logger.info("Initialized StructuredReportGenerator with safety checks: %s", safety_checks)

    def generate_pdf_report(self, report: MedicalReport, output_path: str) -> None:
        """
        Generate a formatted PDF report from the medical report data.
        
        Args:
            report: MedicalReport object containing the report data
            output_path: Path where the PDF should be saved
            
        Returns:
            None
        """
        try:
            # Initialize PDF document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # Get styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=16,
                spaceAfter=30
            )
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=12
            )
            normal_style = styles['Normal']
            
            # Build document content
            content = []
            
            # Title
            content.append(Paragraph('Medical Image Analysis Report', title_style))
            content.append(Spacer(1, 12))
            
            # Date and Time
            content.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M:%S')}", normal_style))
            content.append(Spacer(1, 20))
            
            # Clinical Findings Section
            content.append(Paragraph('Clinical Findings', heading_style))
            
            # Image Quality
            if report.clinical_findings.get('image_quality'):
                quality_data = []
                quality = report.clinical_findings['image_quality']
                if quality.get('quality_notes'):
                    quality_data.extend([note for note in quality['quality_notes'] if note])
                if quality.get('technical_factors'):
                    tech = quality['technical_factors']
                    quality_data.append(f"Image dimensions: {tech.get('dimensions', '')}")
                    quality_data.append(f"Image mode: {tech.get('mode', '')}")
                
                for note in quality_data:
                    if note and isinstance(note, str):
                        content.append(Paragraph(note, normal_style))
                        
            content.append(Spacer(1, 12))
            
            # Primary Findings
            if report.clinical_findings.get('observations', {}).get('primary_findings'):
                content.append(Paragraph('Primary Findings:', normal_style))
                for finding in report.clinical_findings['observations']['primary_findings']:
                    if finding and isinstance(finding, str):
                        content.append(Paragraph(f"• {finding}", normal_style))
                        
            content.append(Spacer(1, 20))
            
            # Diagnostic Interpretation Section
            content.append(Paragraph('Diagnostic Interpretation', heading_style))
            
            # Primary Diagnosis
            if report.diagnostic_interpretation.get('primary_diagnosis'):
                diagnosis = report.diagnostic_interpretation['primary_diagnosis']
                content.append(Paragraph('Primary Diagnosis:', normal_style))
                content.append(Paragraph(f"• Condition: {diagnosis.get('condition', 'N/A')}", normal_style))
                content.append(Paragraph(f"• Confidence: {diagnosis.get('confidence', 0)*100:.1f}%", normal_style))
                
                if diagnosis.get('supporting_evidence'):
                    content.append(Paragraph('Supporting Evidence:', normal_style))
                    for evidence in diagnosis['supporting_evidence']:
                        content.append(Paragraph(f"• {evidence}", normal_style))
                        
            content.append(Spacer(1, 20))
            
            # Patient Explanation Section
            content.append(Paragraph('Patient Information', heading_style))
            
            if report.patient_explanation:
                if 'summary' in report.patient_explanation:
                    content.append(Paragraph('Summary:', normal_style))
                    content.append(Paragraph(report.patient_explanation['summary'], normal_style))
                    
                if 'what_this_means' in report.patient_explanation:
                    content.append(Paragraph('What This Means:', normal_style))
                    for item in report.patient_explanation['what_this_means']:
                        content.append(Paragraph(f"• {item}", normal_style))
                        
                if 'next_steps' in report.patient_explanation:
                    content.append(Paragraph('Next Steps:', normal_style))
                    for step in report.patient_explanation['next_steps']:
                        content.append(Paragraph(f"• {step}", normal_style))
                        
            # Build PDF
            doc.build(content)
            logger.info(f"PDF report generated successfully: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate PDF report: {str(e)}")
            raise MedicalReportError(f"PDF report generation failed: {str(e)}")