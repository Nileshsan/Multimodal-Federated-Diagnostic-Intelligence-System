from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'patients', views.PatientViewSet, basename='patient')
router.register(r'images', views.MedicalImageViewSet, basename='medical-image')
router.register(r'diagnoses', views.DiagnosisViewSet, basename='diagnosis')
router.register(r'reports', views.MedicalReportViewSet, basename='medical-report')

urlpatterns = [
    path('', include(router.urls)),
    
    # Analysis endpoints
    path('analyze/image/', views.ImageAnalysis.as_view(), name='analyze-image'),
    path('analyze/text/', views.TextAnalysis.as_view(), name='analyze-text'),
]