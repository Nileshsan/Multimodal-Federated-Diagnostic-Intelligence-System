"""
Medical image processor for handling different types of medical imaging.
"""

import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, Optional, Union, Any
import pydicom  # For DICOM file handling

class ImageProcessor:
    """Processor for medical images including X-rays, CT scans, MRI, etc."""
    
    SUPPORTED_FORMATS = {
        'xray': ['.jpg', '.jpeg', '.png', '.dcm'],
        'ct': ['.dcm', '.nii', '.nii.gz'],
        'mri': ['.dcm', '.nii', '.nii.gz'],
        'ultrasound': ['.jpg', '.jpeg', '.png', '.dcm']
    }
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_image(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Load and preprocess medical image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            dict: Processed image data including:
                - image: Processed image tensor
                - metadata: Image metadata
                - format: Detected image format
        """
        image_path = Path(image_path)
        extension = image_path.suffix.lower()
        
        try:
            metadata = {}
            
            # Handle DICOM files
            if extension == '.dcm':
                ds = pydicom.dcmread(image_path)
                metadata = {
                    'PatientID': getattr(ds, 'PatientID', 'Unknown'),
                    'Modality': getattr(ds, 'Modality', 'Unknown'),
                    'StudyDate': getattr(ds, 'StudyDate', 'Unknown'),
                    'BodyPartExamined': getattr(ds, 'BodyPartExamined', 'Unknown')
                }
                # Convert to image array
                image = ds.pixel_array
                image = Image.fromarray(image)
            
            # Handle regular image files
            else:
                image = Image.open(image_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                metadata = {
                    'format': image.format,
                    'size': image.size,
                    'mode': image.mode
                }
            
            # Determine image format based on content and metadata
            img_format = self._detect_image_format(image_path, metadata)
            
            return {
                'image': image,
                'metadata': metadata,
                'format': img_format
            }
            
        except Exception as e:
            raise ValueError(f"Error loading image {image_path}: {str(e)}")
    
    def _detect_image_format(self, image_path: Path, metadata: Dict) -> str:
        """Detect specific medical image format.
        
        Args:
            image_path: Path to image
            metadata: Image metadata
            
        Returns:
            str: Detected format (xray, ct, mri, ultrasound)
        """
        # Check DICOM metadata first
        if image_path.suffix.lower() == '.dcm':
            modality = metadata.get('Modality', '').upper()
            body_part = metadata.get('BodyPartExamined', '').lower()
            
            if modality == 'CR' or modality == 'DX':
                return 'xray'
            elif modality == 'CT':
                return 'ct'
            elif modality == 'MR':
                return 'mri'
            elif modality == 'US':
                return 'ultrasound'
        
        # For regular images, try to infer from filename and path
        path_str = str(image_path).lower()
        if any(x in path_str for x in ['xray', 'x-ray', 'chest']):
            return 'xray'
        elif 'ct' in path_str:
            return 'ct'
        elif 'mri' in path_str:
            return 'mri'
        elif 'ultrasound' in path_str:
            return 'ultrasound'
        
        # Default to unknown
        return 'unknown'
    
    def preprocess_image(self, image_data: Dict[str, Any], target_size: tuple = (224, 224)) -> torch.Tensor:
        """Preprocess image for model input.
        
        Args:
            image_data: Loaded image data from load_image
            target_size: Target size for resizing
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        image = image_data['image']
        
        # Resize
        if image.size != target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to tensor
        image_tensor = torch.FloatTensor(np.array(image)).unsqueeze(0)
        
        # Normalize
        image_tensor = image_tensor / 255.0
        image_tensor = image_tensor.to(self.device)
        
        return image_tensor
    
    def get_image_info(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant information about the medical image.
        
        Args:
            image_data: Loaded image data
            
        Returns:
            dict: Image information including:
                - format: Image format (xray, ct, etc.)
                - dimensions: Image dimensions
                - patient_info: Patient information (if available)
                - study_info: Study information (if available)
        """
        info = {
            'format': image_data['format'],
            'dimensions': image_data['image'].size
        }
        
        # Extract DICOM-specific information
        metadata = image_data['metadata']
        if isinstance(metadata.get('PatientID'), str):
            info['patient_info'] = {
                'id': metadata.get('PatientID'),
                'study_date': metadata.get('StudyDate'),
                'body_part': metadata.get('BodyPartExamined')
            }
        
        return info