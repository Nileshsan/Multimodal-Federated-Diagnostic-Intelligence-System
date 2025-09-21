from django.db import models
from django.contrib.auth.models import User

class Patient(models.Model):
    """Patient information model."""
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    date_of_birth = models.DateField()
    medical_history = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.get_full_name()} ({self.user.username})"

class MedicalImage(models.Model):
    """Medical image and analysis results."""
    IMAGE_TYPES = [
        ('xray', 'X-Ray'),
        ('ct', 'CT Scan'),
        ('mri', 'MRI'),
        ('ultrasound', 'Ultrasound'),
        ('report', 'Medical Report'),
        ('other', 'Other'),
    ]
    
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='medical_images/%Y/%m/%d/')
    image_type = models.CharField(max_length=20, choices=IMAGE_TYPES)
    analysis_results = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.image_type} - {self.patient} ({self.created_at})"

class Diagnosis(models.Model):
    """Medical diagnosis and recommendations."""
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
    medical_image = models.ForeignKey(MedicalImage, on_delete=models.SET_NULL, null=True, blank=True)
    symptoms = models.TextField()
    diagnosis = models.TextField()
    recommendations = models.TextField()
    severity = models.CharField(max_length=20, choices=[
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('critical', 'Critical'),
    ])
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = 'diagnoses'

    def __str__(self):
        return f"Diagnosis for {self.patient} - {self.created_at}"

class MedicalReport(models.Model):
    """Complete medical report including images, diagnosis, and follow-ups."""
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
    diagnosis = models.ForeignKey(Diagnosis, on_delete=models.CASCADE)
    summary = models.TextField()
    detailed_report = models.TextField()
    follow_up_recommended = models.BooleanField(default=False)
    follow_up_date = models.DateField(null=True, blank=True)
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Report for {self.patient} - {self.created_at}"
