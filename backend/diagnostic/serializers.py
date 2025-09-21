from rest_framework import serializers
from django.contrib.auth.models import User
from medical_diagnostic.models import Patient, MedicalImage, Diagnosis, MedicalReport

class UserSerializer(serializers.ModelSerializer):
    """Serializer for User model."""
    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'first_name', 'last_name')

class PatientSerializer(serializers.ModelSerializer):
    """Serializer for Patient model."""
    user = UserSerializer(read_only=True)
    class Meta:
        model = Patient
        fields = '__all__'

class MedicalImageSerializer(serializers.ModelSerializer):
    """Serializer for MedicalImage model."""
    patient = PatientSerializer(read_only=True)
    patient_id = serializers.IntegerField(write_only=True)
    
    class Meta:
        model = MedicalImage
        fields = '__all__'

class DiagnosisSerializer(serializers.ModelSerializer):
    """Serializer for Diagnosis model."""
    patient = PatientSerializer(read_only=True)
    patient_id = serializers.IntegerField(write_only=True)
    medical_image = MedicalImageSerializer(read_only=True)
    medical_image_id = serializers.IntegerField(write_only=True, required=False)
    created_by = UserSerializer(read_only=True)
    
    class Meta:
        model = Diagnosis
        fields = '__all__'

class MedicalReportSerializer(serializers.ModelSerializer):
    """Serializer for MedicalReport model."""
    patient = PatientSerializer(read_only=True)
    patient_id = serializers.IntegerField(write_only=True)
    diagnosis = DiagnosisSerializer(read_only=True)
    diagnosis_id = serializers.IntegerField(write_only=True)
    created_by = UserSerializer(read_only=True)
    
    class Meta:
        model = MedicalReport
        fields = '__all__'

class ImageAnalysisSerializer(serializers.Serializer):
    """Serializer for image analysis request."""
    image = serializers.ImageField()
    image_type = serializers.ChoiceField(choices=MedicalImage.IMAGE_TYPES)
    patient_id = serializers.IntegerField()
    symptoms = serializers.CharField(required=False, allow_blank=True)

class TextAnalysisSerializer(serializers.Serializer):
    """Serializer for text analysis request."""
    text = serializers.CharField()
    patient_id = serializers.IntegerField()
    medical_image_id = serializers.IntegerField(required=False)