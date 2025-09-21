"""
Views for the diagnostic app.
"""
import sys
import logging
from pathlib import Path
from rest_framework import generics, status, permissions
from rest_framework.viewsets import ModelViewSet
from rest_framework.decorators import action
from rest_framework.views import APIView
from rest_framework.response import Response
from django.conf import settings
from django.shortcuts import get_object_or_404

from medical_diagnostic.models import Patient, MedicalImage, Diagnosis, MedicalReport
from .serializers import (
    PatientSerializer, MedicalImageSerializer, DiagnosisSerializer,
    MedicalReportSerializer, ImageAnalysisSerializer, TextAnalysisSerializer
)

# Add Models directory to Python path if not already added
if str(settings.MODELS_DIR) not in sys.path:
    sys.path.append(str(settings.MODELS_DIR))

# Import processors and models
from processors.workflow_initializer import WorkflowInitializer
from vision_model.medical_inference import MedicalVisionInference
from processors.text_processor import TextProcessor
if str(settings.MODELS_DIR) not in sys.path:
    sys.path.append(str(settings.MODELS_DIR))

from vision_model.medical_inference import MedicalVisionInference
from processors.text_processor import TextProcessor

class PatientViewSet(ModelViewSet):
    """ViewSet for viewing and editing patient information."""
    queryset = Patient.objects.all()
    serializer_class = PatientSerializer
    permission_classes = [permissions.IsAuthenticated]

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    @action(detail=True)
    def medical_images(self, request, pk=None):
        patient = self.get_object()
        images = MedicalImage.objects.filter(patient=patient)
        serializer = MedicalImageSerializer(images, many=True)
        return Response(serializer.data)

    @action(detail=True)
    def diagnoses(self, request, pk=None):
        patient = self.get_object()
        diagnoses = Diagnosis.objects.filter(patient=patient)
        serializer = DiagnosisSerializer(diagnoses, many=True)
        return Response(serializer.data)

    @action(detail=True)
    def reports(self, request, pk=None):
        patient = self.get_object()
        reports = MedicalReport.objects.filter(patient=patient)
        serializer = MedicalReportSerializer(reports, many=True)
        return Response(serializer.data)

class MedicalImageViewSet(ModelViewSet):
    """ViewSet for viewing and editing medical images."""
    queryset = MedicalImage.objects.all()
    serializer_class = MedicalImageSerializer
    permission_classes = [permissions.IsAuthenticated]

    def perform_create(self, serializer):
        patient = get_object_or_404(Patient, id=self.request.data['patient_id'])
        serializer.save(patient=patient)

class DiagnosisViewSet(ModelViewSet):
    """ViewSet for viewing and editing diagnoses."""
    queryset = Diagnosis.objects.all()
    serializer_class = DiagnosisSerializer
    permission_classes = [permissions.IsAuthenticated]

    def perform_create(self, serializer):
        patient = get_object_or_404(Patient, id=self.request.data['patient_id'])
        if 'medical_image_id' in self.request.data:
            medical_image = get_object_or_404(MedicalImage, id=self.request.data['medical_image_id'])
        else:
            medical_image = None
        serializer.save(
            patient=patient,
            medical_image=medical_image,
            created_by=self.request.user
        )

class MedicalReportViewSet(ModelViewSet):
    """ViewSet for viewing and editing medical reports."""
    queryset = MedicalReport.objects.all()
    serializer_class = MedicalReportSerializer
    permission_classes = [permissions.IsAuthenticated]

    def perform_create(self, serializer):
        patient = get_object_or_404(Patient, id=self.request.data['patient_id'])
        diagnosis = get_object_or_404(Diagnosis, id=self.request.data['diagnosis_id'])
        serializer.save(
            patient=patient,
            diagnosis=diagnosis,
            created_by=self.request.user
        )

class ImageAnalysis(APIView):
    """API endpoint for analyzing medical images with intelligent workflow initialization."""
    permission_classes = [permissions.IsAuthenticated]
    
    def post(self, request):
        serializer = ImageAnalysisSerializer(data=request.data)
        if serializer.is_valid():
            try:
                # Get patient
                patient = get_object_or_404(Patient, id=serializer.validated_data['patient_id'])
                
                # Save uploaded image
                image = serializer.validated_data['image']
                image_type = serializer.validated_data['image_type']
                symptoms = serializer.validated_data.get('symptoms', '')
                
                # Save medical image
                medical_image = MedicalImage.objects.create(
                    patient=patient,
                    image=image,
                    image_type=image_type
                )
                
                # Initialize workflow
                workflow_initializer = WorkflowInitializer(
                    models_dir=settings.MODELS_DIR,
                    logger=logging.getLogger('medical_vision')
                )
                
                # Analyze requirements and determine optimal models
                workflow_analysis = workflow_initializer.analyze_input_requirements(
                    image_path=image.temporary_file_path(),
                    text_description=symptoms
                )
                
                # Initialize main vision model with configuration based on workflow analysis
                vision_model = MedicalVisionInference(
                    model_path=settings.VISION_MODEL_PATH,
                    log_file=settings.MEDIA_ROOT / 'logs' / 'medical_vision.log',
                    config=self._create_model_config(workflow_analysis)
                )
                
                # Analyze image with context from workflow analysis
                analysis = vision_model.analyze_medical_image(
                    image_path=image.temporary_file_path(),
                    prompt=self._create_enhanced_prompt(symptoms, workflow_analysis),
                    context=workflow_analysis
                )
                
                # Update medical image with analysis results
                medical_image.analysis_results = analysis
                medical_image.save()
                
                # Create diagnosis
                diagnosis = Diagnosis.objects.create(
                    patient=patient,
                    medical_image=medical_image,
                    symptoms=symptoms,
                    diagnosis=analysis.get('diagnosis', ''),
                    recommendations=analysis.get('recommendations', ''),
                    severity=analysis.get('severity', 'medium'),
                    created_by=request.user
                )
                
                # Return combined response with workflow analysis
                return Response({
                    'medical_image': MedicalImageSerializer(medical_image).data,
                    'diagnosis': DiagnosisSerializer(diagnosis).data,
                    'analysis': analysis,
                    'workflow_analysis': workflow_analysis
                })
    
    def _create_model_config(self, workflow_analysis):
        """Create model configuration based on workflow analysis."""
        return {
            'image_type': workflow_analysis['image_type'],
            'preprocessing_steps': workflow_analysis['processing_requirements']['preprocessing_steps'],
            'model_priority': workflow_analysis['processing_requirements']['priority_order'],
            'gpu_requirements': workflow_analysis['processing_requirements']['gpu_memory'],
            'confidence_thresholds': {
                'image_quality': workflow_analysis['confidence_scores']['image_quality'],
                'condition_relevance': workflow_analysis['confidence_scores']['condition_relevance']
            }
        }
    
    def _create_enhanced_prompt(self, symptoms, workflow_analysis):
        """Create an enhanced prompt using workflow analysis results."""
        prompt_parts = []
        
        # Add symptom description
        if symptoms:
            prompt_parts.append(f"Given the symptoms: {symptoms}")
        
        # Add image type and characteristics
        prompt_parts.append(f"Analyzing {workflow_analysis['image_type']} image "
                          f"of the {workflow_analysis['anatomical_location']}")
        
        # Add potential conditions to look for
        if workflow_analysis['potential_conditions']:
            conditions = ", ".join(workflow_analysis['potential_conditions'])
            prompt_parts.append(f"Please examine for potential signs of: {conditions}")
        
        # Add specific characteristics to focus on
        if 'image_characteristics' in workflow_analysis:
            prompt_parts.append("Pay special attention to regions with: " + 
                              ", ".join(str(k) for k in workflow_analysis['image_characteristics'].keys()))
        
        return " ".join(prompt_parts)
                
            except Exception as e:
                return Response(
                    {'error': str(e)},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class TextAnalysis(APIView):
    """API endpoint for analyzing medical text."""
    permission_classes = [permissions.IsAuthenticated]
    
    def post(self, request):
        serializer = TextAnalysisSerializer(data=request.data)
        if serializer.is_valid():
            try:
                text = serializer.validated_data['text']
                patient = get_object_or_404(Patient, id=serializer.validated_data['patient_id'])
                medical_image = None
                if 'medical_image_id' in serializer.validated_data:
                    medical_image = get_object_or_404(MedicalImage, id=serializer.validated_data['medical_image_id'])
                
                # Initialize text processor
                text_processor = TextProcessor()
                
                # Analyze text
                analysis = text_processor.analyze_medical_text(text)
                
                # Create diagnosis
                diagnosis = Diagnosis.objects.create(
                    patient=patient,
                    medical_image=medical_image,
                    symptoms=text,
                    diagnosis=analysis.get('diagnosis', ''),
                    recommendations=analysis.get('recommendations', ''),
                    severity=analysis.get('severity', 'medium'),
                    created_by=request.user
                )
                
                # Return combined response
                return Response({
                    'diagnosis': DiagnosisSerializer(diagnosis).data,
                    'analysis': analysis
                })
                
            except Exception as e:
                return Response(
                    {'error': str(e)},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)