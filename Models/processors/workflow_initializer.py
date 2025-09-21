"""
Workflow Initializer for Medical Diagnostic System
Handles initial analysis of inputs and determines appropriate models to use.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import torch
import numpy as np
from transformers import AutoFeatureExtractor
from .image_processor import ImageProcessor
from .text_processor import TextProcessor

class WorkflowInitializer:
    """Initializes the diagnostic workflow based on input analysis."""
    
    # Model type definitions
    MODEL_TYPES = {
        'xray': {
            'chest': ['densenet121', 'resnet50', 'efficientnet-b4'],
            'bone': ['inception-v3', 'resnet101', 'densenet169'],
            'abdominal': ['efficientnet-b3', 'densenet161', 'vgg19']
        },
        'ct': {
            'brain': ['3d-resnet', 'densenet3d', 'mednet'],
            'lung': ['3d-efficientnet', 'resnet3d-50', 'convnext-3d'],
            'abdominal': ['3d-unet', 'vnet', 'densenet3d-121']
        },
        'mri': {
            'brain': ['3d-swin', 'vit-3d', 'medformer'],
            'spine': ['3d-resnet101', 'transformer3d', 'efficient3d'],
            'joint': ['3d-inception', 'resnet3d-152', 'med3d']
        }
    }
    
    # Common conditions and their associated image characteristics
    CONDITION_FEATURES = {
        'pneumonia': ['opacity', 'consolidation', 'infiltrates'],
        'fracture': ['bone_discontinuity', 'alignment', 'density'],
        'tumor': ['mass', 'nodule', 'enhancement'],
        'inflammation': ['swelling', 'fluid', 'density_change']
    }

    def __init__(
        self,
        models_dir: Union[str, Path],
        device: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """Initialize the workflow initializer.
        
        Args:
            models_dir: Directory containing model files
            device: Device to use for computation ('cuda' or 'cpu')
            logger: Logger instance for tracking
        """
        self.models_dir = Path(models_dir)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize processors
        self.image_processor = ImageProcessor()
        self.text_processor = TextProcessor()
        
        # Load feature extractors for initial analysis
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            'microsoft/resnet-50',
            cache_dir=self.models_dir / 'cache'
        )

    def analyze_input_requirements(
        self,
        image_path: Union[str, Path],
        text_description: str
    ) -> Dict:
        """Analyze inputs to determine required models and processing pipeline.
        
        Args:
            image_path: Path to the medical image
            text_description: Description of symptoms or medical context
            
        Returns:
            Dict containing analysis results and recommended models
        """
        self.logger.info("Starting input analysis...")
        
        # Load and analyze image
        image_type, image_characteristics = self._analyze_image(image_path)
        
        # Analyze text description
        conditions, anatomical_location = self._analyze_text(text_description)
        
        # Determine optimal models
        recommended_models = self._determine_models(
            image_type,
            anatomical_location,
            conditions
        )
        
        # Calculate confidence scores
        confidence_scores = self._calculate_confidence_scores(
            image_characteristics,
            conditions
        )
        
        analysis_result = {
            'image_type': image_type,
            'image_characteristics': image_characteristics,
            'anatomical_location': anatomical_location,
            'potential_conditions': conditions,
            'recommended_models': recommended_models,
            'confidence_scores': confidence_scores,
            'processing_requirements': {
                'gpu_memory': self._estimate_gpu_requirements(recommended_models),
                'preprocessing_steps': self._determine_preprocessing_steps(image_type),
                'priority_order': self._determine_model_priority(confidence_scores)
            }
        }
        
        self.logger.info(f"Input analysis completed: {len(recommended_models)} models recommended")
        return analysis_result

    def _analyze_image(
        self,
        image_path: Union[str, Path]
    ) -> Tuple[str, Dict]:
        """Analyze the image to determine its type and characteristics."""
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            image_array = np.array(image)
            
            # Determine image type based on characteristics
            image_type = self._determine_image_type(image_array)
            
            # Extract image characteristics
            characteristics = {
                'dimensions': image_array.shape,
                'intensity_range': (float(image_array.min()), float(image_array.max())),
                'mean_intensity': float(image_array.mean()),
                'std_intensity': float(image_array.std()),
                'histogram': np.histogram(image_array, bins=50)[0].tolist(),
                'spatial_features': self._extract_spatial_features(image_array)
            }
            
            return image_type, characteristics
            
        except Exception as e:
            self.logger.error(f"Error analyzing image: {str(e)}")
            raise

    def _analyze_text(
        self,
        text: str
    ) -> Tuple[List[str], str]:
        """Analyze the text description to extract conditions and anatomical location."""
        try:
            # Use text processor to extract key information
            processed_text = self.text_processor.analyze_medical_text(text)
            
            # Extract potential conditions
            conditions = processed_text.get('conditions', [])
            
            # Extract anatomical location
            location = processed_text.get('anatomical_location', 'unknown')
            
            return conditions, location
            
        except Exception as e:
            self.logger.error(f"Error analyzing text: {str(e)}")
            raise

    def _determine_models(
        self,
        image_type: str,
        anatomical_location: str,
        conditions: List[str]
    ) -> List[Dict]:
        """Determine which models to use based on analysis results."""
        recommended_models = []
        
        # Get base models for image type and anatomical location
        if image_type in self.MODEL_TYPES:
            if anatomical_location in self.MODEL_TYPES[image_type]:
                base_models = self.MODEL_TYPES[image_type][anatomical_location]
                recommended_models.extend([
                    {
                        'name': model,
                        'type': 'primary',
                        'purpose': f'{image_type}_{anatomical_location}_analysis'
                    }
                    for model in base_models[:2]  # Use top 2 models
                ])
        
        # Add condition-specific models
        for condition in conditions:
            condition_models = self._get_condition_specific_models(
                condition,
                image_type,
                anatomical_location
            )
            recommended_models.extend(condition_models)
        
        return recommended_models

    def _calculate_confidence_scores(
        self,
        image_characteristics: Dict,
        conditions: List[str]
    ) -> Dict:
        """Calculate confidence scores for different analysis aspects."""
        confidence_scores = {
            'image_quality': self._assess_image_quality(image_characteristics),
            'condition_relevance': {}
        }
        
        # Calculate relevance scores for each condition
        for condition in conditions:
            if condition in self.CONDITION_FEATURES:
                expected_features = self.CONDITION_FEATURES[condition]
                relevance_score = self._calculate_feature_relevance(
                    image_characteristics,
                    expected_features
                )
                confidence_scores['condition_relevance'][condition] = relevance_score
        
        return confidence_scores

    def _extract_spatial_features(self, image_array: np.ndarray) -> Dict:
        """Extract spatial features from the image."""
        return {
            'gradient_magnitude': float(np.gradient(image_array).mean()),
            'edge_density': float(self._calculate_edge_density(image_array)),
            'texture_features': self._calculate_texture_features(image_array)
        }

    def _determine_image_type(self, image_array: np.ndarray) -> str:
        """Determine the type of medical image based on its characteristics."""
        # Implementation based on image characteristics
        # This is a simplified version - in practice, would use more sophisticated methods
        if len(image_array.shape) == 2:
            return 'xray'
        elif len(image_array.shape) == 3:
            if image_array.shape[-1] == 1:
                return 'ct'
            else:
                return 'mri'
        return 'unknown'

    def _get_condition_specific_models(
        self,
        condition: str,
        image_type: str,
        anatomical_location: str
    ) -> List[Dict]:
        """Get condition-specific models based on the analysis."""
        # Implementation would include logic to select specialized models
        # based on specific conditions
        return []

    def _calculate_feature_relevance(
        self,
        characteristics: Dict,
        expected_features: List[str]
    ) -> float:
        """Calculate relevance score based on image characteristics."""
        # Implementation would include comparison of image features
        # with expected features for specific conditions
        return 0.75  # Placeholder

    def _assess_image_quality(self, characteristics: Dict) -> float:
        """Assess the quality of the input image."""
        # Implementation would include various quality metrics
        return 0.85  # Placeholder

    def _estimate_gpu_requirements(
        self,
        recommended_models: List[Dict]
    ) -> Dict:
        """Estimate GPU memory requirements for recommended models."""
        return {
            'minimum_memory': '4GB',
            'recommended_memory': '8GB',
            'can_run_parallel': True
        }

    def _determine_preprocessing_steps(self, image_type: str) -> List[str]:
        """Determine required preprocessing steps."""
        common_steps = ['normalization', 'noise_reduction']
        type_specific_steps = {
            'xray': ['contrast_enhancement', 'artifact_removal'],
            'ct': ['slice_selection', '3d_reconstruction'],
            'mri': ['bias_field_correction', 'registration']
        }
        return common_steps + type_specific_steps.get(image_type, [])