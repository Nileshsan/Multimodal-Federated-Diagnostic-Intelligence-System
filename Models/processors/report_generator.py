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
import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from scipy import stats
import numpy as np
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

class FindingSeverity(Enum):
    """Enumeration for finding severity levels."""
    CRITICAL = auto()
    SEVERE = auto()
    MODERATE = auto()
    MILD = auto()
    NORMAL = auto()

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
        
    def _load_knowledge_base(self, path: Optional[Path]) -> Dict:
        """
        Load and validate medical knowledge base.
        
        Args:
            path: Path to knowledge base JSON file
            
        Returns:
            Dict: Validated knowledge base structure
            
        Raises:
            KnowledgeBaseError: If knowledge base loading or validation fails
        """
        if not path:
            logger.warning("No knowledge base provided, using default medical knowledge")
            return self._get_default_knowledge_base()
            
        try:
            with open(path, 'r', encoding='utf-8') as f:
                kb = json.load(f)
            
            # Validate knowledge base structure
            self._validate_knowledge_base_structure(kb)
            return kb
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("Failed to load knowledge base: %s. Using default.", str(e))
            return self._get_default_knowledge_base()
    
    def _validate_knowledge_base_structure(self, kb: Dict) -> None:
        """Validate that knowledge base has required structure."""
        required_sections = ['conditions', 'observations', 'measurements', 'critical_conditions']
        missing = [s for s in required_sections if s not in kb]
        if missing:
            raise KnowledgeBaseError(f"Missing required sections: {missing}")
    
    def _get_default_knowledge_base(self) -> Dict:
        """Return a default medical knowledge base for basic functionality."""
        return {
            'conditions': {
                'pneumonia': {
                    'symptoms': ['consolidation', 'opacity', 'infiltrates'],
                    'severity': 'moderate',
                    'base_confidence': 0.7,
                    'description': 'Lung infection with inflammation'
                },
                'normal': {
                    'symptoms': ['clear lungs', 'normal heart size'],
                    'severity': 'normal',
                    'base_confidence': 0.9,
                    'description': 'No significant abnormalities detected'
                }
            },
            'observations': {
                'opacity': {
                    'explanation': 'Area of increased density in lung tissue',
                    'implications': ['possible infection', 'inflammation'],
                    'severity': 'moderate'
                },
                'consolidation': {
                    'explanation': 'Complete filling of lung air spaces',
                    'implications': ['pneumonia', 'lung collapse'],
                    'severity': 'moderate'
                }
            },
            'measurements': {
                'heart_size': {
                    'normal_range': {'min': 0.45, 'max': 0.55},
                    'units': 'ratio'
                }
            },
            'critical_conditions': ['massive pneumothorax', 'tension pneumothorax', 'large pleural effusion'],
            'lay_terms': {
                'opacity': 'cloudy area',
                'consolidation': 'filled air spaces',
                'infiltrates': 'abnormal substances'
            }
        }
        
    def generate_structured_report(
        self,
        model_output: str,
        image_analysis: Dict,
        patient_context: Optional[Dict] = None
    ) -> MedicalReport:
        """
        Generate a validated structured medical report.
        
        Args:
            model_output: Raw model analysis output
            image_analysis: Image processing results
            patient_context: Optional patient medical context
            
        Returns:
            MedicalReport: Comprehensive validated report
            
        Raises:
            ValidationError: If report validation fails
            MedicalReportError: For other report generation errors
        """
        try:
            # Validate inputs
            if not model_output or not model_output.strip():
                raise ValidationError("Model output is empty or invalid")
            if not image_analysis or not isinstance(image_analysis, dict):
                raise ValidationError("Image analysis data is missing or invalid")
            
            # Extract and validate findings
            findings = self._extract_findings(model_output)
            if not any(findings.values()):
                raise ValidationError("No findings extracted from model output")
            
            # Enhance with medical knowledge
            enhanced_findings = self._enhance_with_medical_knowledge(findings)
            
            # Generate report components
            report = MedicalReport(
                clinical_findings=self._generate_clinical_findings(
                    enhanced_findings,
                    image_analysis
                ),
                diagnostic_interpretation=self._generate_diagnostic_interpretation(
                    enhanced_findings,
                    patient_context
                ),
                technical_details=self._generate_technical_details(
                    image_analysis
                ),
                patient_explanation=self._generate_patient_explanation(
                    enhanced_findings
                ),
                additional_notes=self._generate_additional_notes(
                    enhanced_findings,
                    image_analysis
                ),
                metadata=self._generate_metadata()
            )
            
            # Validate the complete report
            if self.safety_checks:
                self._validate_report(report)
            
            logger.info("Successfully generated medical report")
            return report
            
        except Exception as e:
            logger.error("Report generation failed: %s", str(e))
            raise MedicalReportError(f"Failed to generate report: {str(e)}")
    
    def _extract_findings(self, model_output: str) -> Dict:
        """Extract structured findings from model output using pattern matching."""
        findings = {
            'primary_observations': [],
            'anatomical_features': [],
            'abnormalities': [],
            'measurements': [],
            'comparisons': [],
            'image_quality_notes': []
        }
        
        try:
            # Clean the input
            text = model_output.strip()
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            # Pattern matching for different types of findings
            patterns = {
                'findings': r'(?:finding|observation|noted?):?\s*(.+?)(?:\.|$)',
                'abnormal': r'(?:abnormal|pathologic|concerning):?\s*(.+?)(?:\.|$)',
                'anatomy': r'(?:heart|lung|bone|rib|diaphragm|mediastin):?\s*(.+?)(?:\.|$)',
                'measurement': r'(\d+(?:\.\d+)?)\s*(?:mm|cm|ratio|percent|%)',
                'quality': r'(?:quality|technique|positioning):?\s*(.+?)(?:\.|$)'
            }
            
            for line in lines:
                line_lower = line.lower()
                
                # Extract findings based on patterns
                for pattern_type, pattern in patterns.items():
                    matches = re.findall(pattern, line_lower, re.IGNORECASE)
                    for match in matches:
                        if pattern_type == 'findings':
                            findings['primary_observations'].append(match.strip())
                        elif pattern_type == 'abnormal':
                            findings['abnormalities'].append(match.strip())
                        elif pattern_type == 'anatomy':
                            findings['anatomical_features'].append(match.strip())
                        elif pattern_type == 'measurement':
                            findings['measurements'].append({
                                'value': float(match),
                                'text': line.strip()
                            })
                        elif pattern_type == 'quality':
                            findings['image_quality_notes'].append(match.strip())
                
                # Look for bullet points or numbered lists
                if line.startswith(('-', '*', '•')) or re.match(r'^\d+\.', line):
                    item = re.sub(r'^[-*•\d\.]\s*', '', line).strip()
                    if any(word in line_lower for word in ['abnormal', 'pathologic', 'concerning']):
                        findings['abnormalities'].append(item)
                    else:
                        findings['primary_observations'].append(item)
            
            # Remove duplicates while preserving order
            for key in findings:
                if isinstance(findings[key], list):
                    findings[key] = list(dict.fromkeys(findings[key]))
            
            return findings
            
        except Exception as e:
            logger.error("Failed to extract findings: %s", str(e))
            raise
    
    def _enhance_with_medical_knowledge(self, findings: Dict) -> Dict:
        """
        Enhance findings with medical knowledge base information.
        
        Args:
            findings: Dictionary of extracted findings
            
        Returns:
            Dict: Enhanced findings with medical context
        """
        enhanced_findings = findings.copy()
        enhanced_findings['medical_context'] = []
        
        try:
            # Enhance primary observations
            for observation in findings['primary_observations']:
                context = self._get_observation_context(observation)
                if context:
                    enhanced_findings['medical_context'].append(context)
            
            # Enhance abnormalities
            for abnormality in findings['abnormalities']:
                context = self._get_observation_context(abnormality)
                if context:
                    enhanced_findings['medical_context'].append(context)
            
            return enhanced_findings
            
        except Exception as e:
            logger.error("Failed to enhance findings: %s", str(e))
            return findings
    
    def _get_observation_context(self, observation: str) -> Optional[Dict]:
        """Get medical context for a specific observation."""
        observation_lower = observation.lower()
        
        # Check direct matches in knowledge base
        for kb_obs, info in self.knowledge_base.get('observations', {}).items():
            if kb_obs.lower() in observation_lower or observation_lower in kb_obs.lower():
                return {
                    'observation': observation,
                    'explanation': info.get('explanation', ''),
                    'implications': info.get('implications', []),
                    'severity': info.get('severity', 'unknown'),
                    'confidence': info.get('base_confidence', 0.5)
                }
        
        return None
    
    def _generate_clinical_findings(self, findings: Dict, image_analysis: Dict) -> Dict:
        """Generate detailed clinical findings section."""
        return {
            'observations': {
                'primary_findings': findings.get('primary_observations', []),
                'anatomical_features': findings.get('anatomical_features', []),
                'abnormalities': findings.get('abnormalities', [])
            },
            'measurements': {
                'quantitative_data': findings.get('measurements', []),
                'reference_values': self._get_reference_values(findings.get('measurements', []))
            },
            'image_quality': {
                'technical_factors': image_analysis.get('image_quality', {}),
                'quality_notes': findings.get('image_quality_notes', [])
            },
            'severity_assessment': self._assess_finding_severity(findings)
        }
    
    def _assess_finding_severity(self, findings: Dict) -> Dict:
        """Assess the overall severity of findings."""
        severity_scores = []
        
        # Check abnormalities against critical conditions
        for abnormality in findings.get('abnormalities', []):
            if abnormality.lower() in [cond.lower() for cond in self.knowledge_base.get('critical_conditions', [])]:
                severity_scores.append(4)  # Critical
            else:
                severity_scores.append(2)  # Moderate
        
        # Default to normal if no abnormalities
        if not severity_scores:
            severity_scores = [0]  # Normal
        
        avg_severity = sum(severity_scores) / len(severity_scores)
        
        if avg_severity >= 3.5:
            level = "Critical"
        elif avg_severity >= 2.5:
            level = "Severe"
        elif avg_severity >= 1.5:
            level = "Moderate"
        elif avg_severity >= 0.5:
            level = "Mild"
        else:
            level = "Normal"
        
        return {
            'overall_severity': level,
            'severity_score': avg_severity,
            'requires_urgent_attention': avg_severity >= 3.5
        }
    
    def _get_reference_values(self, measurements: List) -> Dict:
        """Get reference values for measurements."""
        references = {}
        for measurement in measurements:
            if isinstance(measurement, dict) and 'value' in measurement:
                # Simple reference lookup
                references[measurement['text']] = {
                    'normal_range': 'Within normal limits',
                    'interpretation': 'Normal' if 0.4 <= measurement['value'] <= 0.6 else 'Abnormal'
                }
        return references
    
    def _generate_diagnostic_interpretation(self, findings: Dict, patient_context: Optional[Dict]) -> Dict:
        """Generate diagnostic interpretation with confidence levels."""
        # Determine primary diagnosis
        primary_diagnosis = self._determine_primary_diagnosis(findings)
        
        return {
            'primary_diagnosis': primary_diagnosis,
            'differential_diagnoses': self._generate_differential_diagnoses(findings, primary_diagnosis),
            'confidence_assessment': {
                'overall_confidence': primary_diagnosis.get('confidence', 0.5),
                'confidence_level': self._get_confidence_level(primary_diagnosis.get('confidence', 0.5)),
                'factors_affecting_confidence': self._get_confidence_factors(findings)
            },
            'clinical_correlation': self._correlate_with_patient_context(primary_diagnosis, patient_context) if patient_context else None
        }
    
    def _determine_primary_diagnosis(self, findings: Dict) -> Dict:
        """Determine the primary diagnosis based on findings."""
        diagnosis = {
            'condition': 'Normal study',
            'confidence': 0.8,
            'supporting_evidence': [],
            'icd_code': None,
            'description': 'No significant abnormalities detected'
        }
        
        # Check for abnormalities
        abnormalities = findings.get('abnormalities', [])
        if abnormalities:
            # Simple matching against known conditions
            max_confidence = 0
            best_match = None
            
            for condition, info in self.knowledge_base.get('conditions', {}).items():
                symptoms = info.get('symptoms', [])
                matches = sum(1 for symptom in symptoms if any(symptom.lower() in abnorm.lower() for abnorm in abnormalities))
                
                if matches > 0:
                    confidence = min(0.9, info.get('base_confidence', 0.5) + (matches * 0.1))
                    if confidence > max_confidence:
                        max_confidence = confidence
                        best_match = {
                            'condition': condition,
                            'confidence': confidence,
                            'supporting_evidence': abnormalities[:3],  # Limit to 3 pieces of evidence
                            'description': info.get('description', f'Findings consistent with {condition}')
                        }
            
            if best_match:
                diagnosis = best_match
        
        return diagnosis
    
    def _get_confidence_level(self, confidence: float) -> DiagnosticConfidence:
        """Convert numeric confidence to enum."""
        if confidence >= 0.9:
            return DiagnosticConfidence.HIGH
        elif confidence >= 0.7:
            return DiagnosticConfidence.MODERATE
        elif confidence >= 0.5:
            return DiagnosticConfidence.LOW
        else:
            return DiagnosticConfidence.UNCERTAIN
    
    def _get_confidence_factors(self, findings: Dict) -> List[str]:
        """Identify factors affecting diagnostic confidence."""
        factors = []
        
        if not findings.get('primary_observations'):
            factors.append("Limited clinical findings available")
        
        if findings.get('image_quality_notes'):
            factors.append("Image quality may affect interpretation")
        
        if len(findings.get('abnormalities', [])) > 3:
            factors.append("Multiple abnormalities present")
        
        return factors
    
    def _generate_differential_diagnoses(self, findings: Dict, primary_diagnosis: Dict) -> List[Dict]:
        """Generate list of differential diagnoses."""
        differentials = []
        primary_condition = primary_diagnosis.get('condition', '')
        
        # Simple differential generation based on overlapping symptoms
        for condition, info in self.knowledge_base.get('conditions', {}).items():
            if condition == primary_condition:
                continue
            
            symptoms = info.get('symptoms', [])
            abnormalities = findings.get('abnormalities', [])
            
            matches = sum(1 for symptom in symptoms if any(symptom.lower() in abnorm.lower() for abnorm in abnormalities))
            
            if matches > 0:
                confidence = min(0.8, info.get('base_confidence', 0.3) + (matches * 0.1))
                differentials.append({
                    'condition': condition,
                    'confidence': confidence,
                    'supporting_findings': [abnorm for abnorm in abnormalities if any(symptom.lower() in abnorm.lower() for symptom in symptoms)]
                })
        
        # Sort by confidence and return top 3
        differentials.sort(key=lambda x: x['confidence'], reverse=True)
        return differentials[:3]
    
    def _correlate_with_patient_context(self, diagnosis: Dict, patient_context: Dict) -> Dict:
        """Correlate findings with patient medical context."""
        correlation = {
            'relevant_history': [],
            'risk_factors': [],
            'supporting_factors': [],
            'contradicting_factors': []
        }
        
        # This would be expanded based on the patient context structure
        if patient_context.get('age', 0) > 65:
            correlation['risk_factors'].append('Advanced age')
        
        if patient_context.get('smoking_history'):
            correlation['risk_factors'].append('Smoking history')
        
        return correlation
    
    def _generate_technical_details(self, image_analysis: Dict) -> Dict:
        """Generate technical details about the imaging and analysis."""
        return {
            'imaging_parameters': {
                'modality': image_analysis.get('image_type', 'X-ray'),
                'dimensions': image_analysis.get('image_quality', {}).get('dimensions', 'Unknown'),
                'technical_quality': self._assess_technical_quality(image_analysis)
            },
            'analysis_details': {
                'processing_time': image_analysis.get('processing_time', 0),
                'model_version': 'Llama 3.2 11B Vision',
                'analysis_timestamp': datetime.now().isoformat()
            },
            'quality_metrics': {
                'image_sharpness': 'Adequate',
                'contrast': 'Appropriate',
                'positioning': 'Standard'
            }
        }
    
    def _assess_technical_quality(self, image_analysis: Dict) -> str:
        """Assess technical quality of the image."""
        quality_factors = []
        
        # Check image dimensions
        dimensions = image_analysis.get('image_quality', {}).get('dimensions', (0, 0))
        if isinstance(dimensions, tuple) and len(dimensions) == 2:
            if min(dimensions) < 512:
                quality_factors.append('Low resolution')
            elif min(dimensions) > 1024:
                quality_factors.append('High resolution')
        
        # Overall assessment
        if not quality_factors:
            return 'Good'
        elif any('Low' in factor for factor in quality_factors):
            return 'Limited'
        else:
            return 'Excellent'
    
    def _generate_patient_explanation(self, findings: Dict) -> Dict:
        """Generate patient-friendly explanation of findings."""
        return {
            'summary': self._create_patient_summary(findings),
            'simplified_findings': self._simplify_medical_terms(findings),
            'what_this_means': self._explain_implications(findings),
            'next_steps': self._suggest_next_steps(findings),
            'when_to_seek_care': self._when_to_seek_care(findings)
        }
    
    def _create_patient_summary(self, findings: Dict) -> str:
        """Create a simple summary for patients."""
        abnormalities = findings.get('abnormalities', [])
        
        if not abnormalities:
            return "Your X-ray appears normal with no significant abnormalities detected."
        elif len(abnormalities) == 1:
            return f"Your X-ray shows {abnormalities[0]}. Your doctor will explain what this means for you."
        else:
            return f"Your X-ray shows some findings that your doctor will discuss with you."
    
    def _simplify_medical_terms(self, findings: Dict) -> Dict:
        """Convert medical terms to patient-friendly language."""
        simplified = {}
        lay_terms = self.knowledge_base.get('lay_terms', {})
        
        for observation in findings.get('primary_observations', []):
            simple_term = lay_terms.get(observation.lower(), observation)
            simplified[simple_term] = f"Area of concern that requires medical evaluation"
        
        return simplified
    
    def _explain_implications(self, findings: Dict) -> List[str]:
        """Explain what the findings might mean."""
        implications = []
        
        if not findings.get('abnormalities'):
            implications.append("No immediate medical concerns identified")
        else:
            implications.append("Some findings require medical interpretation")
            implications.append("Your doctor will explain the significance of these findings")
        
        return implications
    
    def _suggest_next_steps(self, findings: Dict) -> List[str]:
        """Suggest appropriate next steps for the patient."""
        steps = []
        
        severity = self._assess_finding_severity(findings)
        if severity.get('requires_urgent_attention'):
            steps.append("Seek immediate medical attention")
        else:
            steps.append("Follow up with your healthcare provider")
            steps.append("Bring this report to your next appointment")
        
        return steps
    
    def _when_to_seek_care(self, findings: Dict) -> List[str]:
        """Advise when to seek medical care."""
        care_advice = []
        
        severity = self._assess_finding_severity(findings)
        if severity.get('requires_urgent_attention'):
            care_advice.append("Seek immediate medical attention if you have symptoms")
        else:
            care_advice.append("Contact your doctor if symptoms worsen")
            care_advice.append("Follow your doctor's recommendations for follow-up care")
        
        return care_advice
    
    def _generate_additional_notes(self, findings: Dict, image_analysis: Dict) -> Dict:
        """Generate additional notes and recommendations."""
        return {
            'limitations': self._note_limitations(findings, image_analysis),
            'recommendations': self._generate_recommendations(findings),
            'follow_up_suggested': self._suggest_follow_up_timeframe(findings),
            'additional_imaging': self._suggest_additional_imaging(findings)
        }
    
    def _note_limitations(self, findings: Dict, image_analysis: Dict) -> List[str]:
        """Note limitations of the analysis."""
        limitations = []
        
        if not findings.get('primary_observations'):
            limitations.append("Limited clinical findings available for analysis")
        
        limitations.append("This report is based on AI analysis and requires physician review")
        limitations.append("Clinical correlation is recommended")
        
        return limitations
    
    def _generate_recommendations(self, findings: Dict) -> List[str]:
        """Generate clinical recommendations."""
        recommendations = []
        
        if findings.get('abnormalities'):
            recommendations.append("Clinical correlation recommended")
            recommendations.append("Consider follow-up imaging if clinically indicated")
        else:
            recommendations.append("Routine follow-up as clinically appropriate")
        
        return recommendations
    
    def _suggest_follow_up_timeframe(self, findings: Dict) -> str:
        """Suggest appropriate follow-up timeframe."""
        severity = self._assess_finding_severity(findings)
        
        if severity.get('requires_urgent_attention'):
            return "Immediate"
        elif severity.get('overall_severity') in ['Severe', 'Moderate']:
            return "1-2 weeks"
        else:
            return "3-6 months or as clinically indicated"
    
    def _suggest_additional_imaging(self, findings: Dict) -> List[str]:
        """Suggest additional imaging if needed."""
        suggestions = []
        
        abnormalities = findings.get('abnormalities', [])
        if abnormalities:
            suggestions.append("Consider CT chest if clinical suspicion remains high")
        
        return suggestions
    
    def _generate_metadata(self) -> Dict:
        """Generate report metadata."""
        return {
            'report_id': f"MR{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'generator_version': '2.0.0',
            'report_format_version': '3.0',
            'analysis_model': 'Llama 3.2 11B Vision',
            'safety_checks_enabled': self.safety_checks,
            'confidence_threshold': self.confidence_threshold
        }
    
    def _validate_report(self, report: MedicalReport) -> None:
        """
        Validate the medical report for safety and completeness.
        
        Args:
            report: The medical report to validate
            
        Raises:
            ValidationError: If validation fails
        """
        # Check required sections
        if not report.clinical_findings.get('observations', {}).get('primary_findings'):
            if not report.clinical_findings.get('observations', {}).get('abnormalities'):
                logger.warning("Report contains no primary findings or abnormalities")
        
        # Validate diagnostic interpretation
        diagnosis = report.diagnostic_interpretation.get('primary_diagnosis', {})
        if not diagnosis.get('condition'):
            raise ValidationError("Missing primary diagnosis")
        
        # Check confidence levels
        confidence = diagnosis.get('confidence', 0)
        if confidence < self.confidence_threshold:
            logger.warning("Low confidence diagnosis: %s (%.2f)", 
                         diagnosis.get('condition'), confidence)
        
        # Check for critical findings
        self._check_critical_findings(report)
        
        # Mark as validated
        report.validation_status['validated'] = True
        report.validation_status['validation_timestamp'] = datetime.now().isoformat()
    
    def _check_critical_findings(self, report: MedicalReport) -> None:
        """Check for critical findings requiring immediate attention."""
        findings = report.clinical_findings.get('observations', {})
        critical_findings = []
        
        # Check abnormalities against critical conditions
        for abnormality in findings.get('abnormalities', []):
            if any(crit.lower() in abnormality.lower() 
                   for crit in self.knowledge_base.get('critical_conditions', [])):
                critical_findings.append(abnormality)
        
        # Check severity assessment
        severity = report.clinical_findings.get('severity_assessment', {})
        if severity.get('requires_urgent_attention'):
            critical_findings.append("High severity findings detected")
        
        if critical_findings:
            logger.warning("Critical findings detected: %s", critical_findings)
            report.validation_status['critical_findings'] = critical_findings
            report.validation_status['requires_urgent_review'] = True

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
            
            # Title and Metadata
            content.append(Paragraph('Medical Image Analysis Report', title_style))
            content.append(Spacer(1, 12))
            
            metadata = report.metadata
            content.append(Paragraph(f"Report ID: {metadata.get('report_id', 'N/A')}", normal_style))
            content.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M:%S')}", normal_style))
            content.append(Paragraph(f"Analysis Model: {metadata.get('analysis_model', 'N/A')}", normal_style))
            content.append(Spacer(1, 20))
            
            # Critical Findings Alert (if any)
            if report.validation_status.get('critical_findings'):
                alert_style = ParagraphStyle(
                    'Alert',
                    parent=styles['Normal'],
                    textColor=colors.red,
                    fontSize=12,
                    spaceAfter=12
                )
                content.append(Paragraph('⚠️ CRITICAL FINDINGS ALERT', alert_style))
                for finding in report.validation_status['critical_findings']:
                    content.append(Paragraph(f"• {finding}", alert_style))
                content.append(Spacer(1, 20))
            
            # Clinical Findings Section
            content.append(Paragraph('Clinical Findings', heading_style))
            
            # Image Quality
            if report.clinical_findings.get('image_quality'):
                quality = report.clinical_findings['image_quality']
                if quality.get('technical_factors'):
                    tech = quality['technical_factors']
                    content.append(Paragraph('Image Technical Details:', normal_style))
                    content.append(Paragraph(f"• Dimensions: {tech.get('dimensions', 'N/A')}", normal_style))
                    content.append(Paragraph(f"• Quality Assessment: {tech.get('quality', 'N/A')}", normal_style))
                
                if quality.get('quality_notes'):
                    content.append(Paragraph('Quality Notes:', normal_style))
                    for note in quality['quality_notes']:
                        if note and isinstance(note, str):
                            content.append(Paragraph(f"• {note}", normal_style))
                            
            content.append(Spacer(1, 12))
            
            # Primary Findings
            observations = report.clinical_findings.get('observations', {})
            if observations.get('primary_findings'):
                content.append(Paragraph('Primary Findings:', normal_style))
                for finding in observations['primary_findings']:
                    if finding and isinstance(finding, str):
                        content.append(Paragraph(f"• {finding}", normal_style))
            
            # Abnormalities
            if observations.get('abnormalities'):
                content.append(Paragraph('Abnormalities:', normal_style))
                for abnormality in observations['abnormalities']:
                    if abnormality and isinstance(abnormality, str):
                        content.append(Paragraph(f"• {abnormality}", normal_style))
                        
            content.append(Spacer(1, 20))
            
            # Diagnostic Interpretation Section
            content.append(Paragraph('Diagnostic Interpretation', heading_style))
            
            # Primary Diagnosis
            diagnosis = report.diagnostic_interpretation.get('primary_diagnosis', {})
            content.append(Paragraph('Primary Diagnosis:', normal_style))
            content.append(Paragraph(f"• Condition: {diagnosis.get('condition', 'N/A')}", normal_style))
            content.append(Paragraph(f"• Confidence: {diagnosis.get('confidence', 0)*100:.1f}%", normal_style))
            
            if diagnosis.get('description'):
                content.append(Paragraph(f"• Description: {diagnosis['description']}", normal_style))
            
            # Differential Diagnoses
            if report.diagnostic_interpretation.get('differential_diagnoses'):
                content.append(Spacer(1, 12))
                content.append(Paragraph('Differential Diagnoses:', normal_style))
                for diff in report.diagnostic_interpretation['differential_diagnoses']:
                    content.append(Paragraph(
                        f"• {diff.get('condition', 'N/A')} "
                        f"(Confidence: {diff.get('confidence', 0)*100:.1f}%)",
                        normal_style
                    ))
                        
            content.append(Spacer(1, 20))
            
            # Patient Information Section
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
                    content.append(Spacer(1, 12))
                    content.append(Paragraph('Next Steps:', normal_style))
                    for step in report.patient_explanation['next_steps']:
                        content.append(Paragraph(f"• {step}", normal_style))
                        
            # Build PDF
            doc.build(content)
            logger.info(f"PDF report generated successfully: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate PDF report: {str(e)}")
            raise MedicalReportError(f"PDF report generation failed: {str(e)}")

