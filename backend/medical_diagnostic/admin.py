from django.contrib import admin
from django.utils.html import format_html
from .models import Patient, MedicalImage, Diagnosis, MedicalReport

@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ('user', 'date_of_birth', 'created_at')
    list_filter = ('created_at', 'updated_at')
    search_fields = ('user__username', 'user__email', 'medical_history')
    date_hierarchy = 'created_at'
    readonly_fields = ('created_at', 'updated_at')

@admin.register(MedicalImage)
class MedicalImageAdmin(admin.ModelAdmin):
    list_display = ('patient', 'image_type', 'image_preview', 'created_at')
    list_filter = ('image_type', 'created_at', 'updated_at')
    search_fields = ('patient__user__username', 'image_type')
    date_hierarchy = 'created_at'
    readonly_fields = ('created_at', 'updated_at', 'image_preview')

    def image_preview(self, obj):
        if obj.image:
            return format_html('<img src="{}" width="100" height="100" />', obj.image.url)
        return 'No Image'
    image_preview.short_description = 'Image Preview'

@admin.register(Diagnosis)
class DiagnosisAdmin(admin.ModelAdmin):
    list_display = ('patient', 'medical_image', 'severity', 'created_by', 'created_at')
    list_filter = ('severity', 'created_at', 'updated_at')
    search_fields = ('patient__user__username', 'symptoms', 'diagnosis', 'recommendations')
    date_hierarchy = 'created_at'
    readonly_fields = ('created_at', 'updated_at')

@admin.register(MedicalReport)
class MedicalReportAdmin(admin.ModelAdmin):
    list_display = ('patient', 'diagnosis', 'follow_up_recommended', 'follow_up_date', 'created_by', 'created_at')
    list_filter = ('follow_up_recommended', 'created_at', 'updated_at')
    search_fields = ('patient__user__username', 'summary', 'detailed_report')
    date_hierarchy = 'created_at'
    readonly_fields = ('created_at', 'updated_at')
