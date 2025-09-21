"""
Text processor for analyzing medical text and symptoms.
"""

import re
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

class TextProcessor:
    """Processor for medical text analysis including symptoms and reports."""
    
    def __init__(self, medical_terms_path: Optional[Path] = None):
        """Initialize text processor.
        
        Args:
            medical_terms_path: Optional path to medical terms dictionary
        """
        self.medical_terms = self._load_medical_terms(medical_terms_path)
        
    def _load_medical_terms(self, path: Optional[Path] = None) -> Dict[str, Any]:
        """Load medical terms dictionary.
        
        Returns:
            dict: Medical terms and their categories
        """
        if path and path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        
        # Default basic medical terms
        return {
            'symptoms': [
                'pain', 'ache', 'fever', 'cough', 'fatigue',
                'nausea', 'vomiting', 'dizziness', 'headache'
            ],
            'anatomical_locations': [
                'chest', 'head', 'arm', 'leg', 'back',
                'neck', 'stomach', 'knee', 'shoulder'
            ],
            'durations': [
                'day', 'days', 'week', 'weeks',
                'month', 'months', 'year', 'years'
            ],
            'severities': [
                'mild', 'moderate', 'severe',
                'slight', 'intense', 'extreme'
            ]
        }
    
    def extract_symptoms(self, text: str) -> Dict[str, Any]:
        """Extract symptoms and their attributes from text.
        
        Args:
            text: Medical text to analyze
            
        Returns:
            dict: Extracted symptoms information
        """
        text = text.lower()
        findings = {
            'symptoms': [],
            'locations': [],
            'durations': [],
            'severities': []
        }
        
        # Extract symptoms
        for symptom in self.medical_terms['symptoms']:
            if symptom in text:
                # Find surrounding context
                pattern = f"\\b{symptom}\\b"
                matches = re.finditer(pattern, text)
                
                for match in matches:
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end]
                    
                    symptom_info = {
                        'name': symptom,
                        'context': context.strip()
                    }
                    findings['symptoms'].append(symptom_info)
        
        # Extract anatomical locations
        for location in self.medical_terms['anatomical_locations']:
            if location in text:
                findings['locations'].append(location)
        
        # Extract duration information
        for duration in self.medical_terms['durations']:
            pattern = f"\\d+\\s+{duration}"
            matches = re.finditer(pattern, text)
            for match in matches:
                findings['durations'].append(match.group())
        
        # Extract severity indicators
        for severity in self.medical_terms['severities']:
            if severity in text:
                findings['severities'].append(severity)
        
        return findings
    
    def analyze_medical_text(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive analysis of medical text.
        
        Args:
            text: Medical text to analyze
            
        Returns:
            dict: Analysis results including:
                - symptoms: Extracted symptoms
                - key_findings: Important medical findings
                - recommendations: Suggested actions
        """
        # Extract symptoms
        symptoms = self.extract_symptoms(text)
        
        # Analyze severity and urgency
        severity_score = self._calculate_severity(symptoms)
        
        # Generate preliminary assessment
        assessment = self._generate_assessment(symptoms, severity_score)
        
        return {
            'symptoms': symptoms,
            'severity_score': severity_score,
            'assessment': assessment
        }
    
    def _calculate_severity(self, symptoms: Dict[str, Any]) -> float:
        """Calculate severity score based on symptoms.
        
        Args:
            symptoms: Extracted symptoms information
            
        Returns:
            float: Severity score (0-1)
        """
        score = 0.0
        max_score = 0.0
        
        # Score based on number of symptoms
        num_symptoms = len(symptoms['symptoms'])
        score += min(num_symptoms * 0.1, 0.5)
        max_score += 0.5
        
        # Score based on severity indicators
        severity_weights = {
            'mild': 0.2,
            'moderate': 0.5,
            'severe': 0.8,
            'extreme': 1.0
        }
        
        for severity in symptoms['severities']:
            score += severity_weights.get(severity, 0.0)
            max_score += 1.0
        
        # Normalize score
        return score / max(max_score, 1.0)
    
    def _generate_assessment(self, symptoms: Dict[str, Any], severity_score: float) -> str:
        """Generate preliminary assessment based on symptoms.
        
        Args:
            symptoms: Extracted symptoms information
            severity_score: Calculated severity score
            
        Returns:
            str: Preliminary assessment text
        """
        assessment = []
        
        # Add symptom summary
        if symptoms['symptoms']:
            symptom_names = [s['name'] for s in symptoms['symptoms']]
            assessment.append(f"Primary symptoms: {', '.join(symptom_names)}")
        
        # Add location information
        if symptoms['locations']:
            assessment.append(f"Affected areas: {', '.join(symptoms['locations'])}")
        
        # Add duration information
        if symptoms['durations']:
            assessment.append(f"Duration: {', '.join(symptoms['durations'])}")
        
        # Add severity assessment
        severity_level = "low"
        if severity_score > 0.7:
            severity_level = "high"
        elif severity_score > 0.4:
            severity_level = "moderate"
        
        assessment.append(f"Overall severity assessment: {severity_level} (score: {severity_score:.2f})")
        
        return "\n".join(assessment)