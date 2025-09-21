"""
Tests for image analysis endpoints.
"""
import os
import tempfile
from django.urls import reverse
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework.test import APITestCase
from rest_framework import status
from medical_diagnostic.models import Patient
from PIL import Image

class ImageAnalysisTests(APITestCase):
    """Test suite for image analysis functionality."""

    def setUp(self):
        # Create test user
        self.user = User.objects.create_user(username='testuser', password='12345')
        
        # Create test patient
        self.patient = Patient.objects.create(
            user=self.user,
            date_of_birth='1990-01-01',
            medical_history='Test history'
        )
        
        # Create a test image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_img:
            img = Image.new('RGB', (100, 100), 'white')
            img.save(tmp_img, format='JPEG')
            self.test_image_path = tmp_img.name
        
        # Login
        self.client.login(username='testuser', password='12345')

    def tearDown(self):
        # Clean up test image
        if os.path.exists(self.test_image_path):
            os.unlink(self.test_image_path)

    def test_image_analysis(self):
        """Test the image analysis endpoint."""
        url = reverse('analyze-image')
        
        # Prepare image data
        with open(self.test_image_path, 'rb') as img:
            data = {
                'image': SimpleUploadedFile(
                    name='test.jpg',
                    content=img.read(),
                    content_type='image/jpeg'
                ),
                'image_type': 'xray',
                'patient_id': self.patient.id,
                'symptoms': 'Test symptoms'
            }
            
            response = self.client.post(url, data, format='multipart')
            
            self.assertEqual(response.status_code, status.HTTP_200_OK)
            self.assertIn('medical_image', response.data)
            self.assertIn('diagnosis', response.data)
            self.assertIn('analysis', response.data)