"""Utility functions for Medical Vision Inference system."""

import time
from typing import Any, Dict, Optional, Tuple
import torch
from PIL import Image
import numpy as np

def setup_gpu_device() -> Tuple[str, Dict[str, str]]:
    """Configure GPU device and memory settings.
    
    Returns:
        Tuple[str, Dict[str, str]]: Device mapping and memory configuration
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this model")
    
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free_mem = torch.cuda.memory_allocated(0) / (1024**3)
    available_mem = gpu_mem - free_mem
    gpu_limit = min(available_mem * 0.9, 7.5)
    
    device_map = "auto"
    max_memory = {
        0: f"{gpu_limit:.1f}GB",
        "cpu": "16GB"
    }
    
    return device_map, max_memory

def validate_image(image: Image.Image) -> Optional[str]:
    """Validate image for processing.
    
    Args:
        image: PIL Image to validate
        
    Returns:
        Optional[str]: Error message if validation fails, None otherwise
    """
    if not isinstance(image, Image.Image):
        return "Input must be a PIL Image"
    
    if image.mode not in ["RGB", "L"]:
        return f"Unsupported image mode: {image.mode}"
    
    min_size = 32
    if image.size[0] < min_size or image.size[1] < min_size:
        return f"Image too small: {image.size}"
    
    return None

def get_aspect_ratio_info(image: Image.Image) -> Tuple[float, int]:
    """Calculate aspect ratio and determine bucket.
    
    Args:
        image: Input image
        
    Returns:
        Tuple[float, int]: Aspect ratio and bucket index
    """
    aspect_ratio = image.size[0] / image.size[1]
    
    if aspect_ratio < 0.75:
        bucket = 1  # portrait
    elif aspect_ratio > 1.33:
        bucket = 2  # landscape
    else:
        bucket = 0  # square-ish
        
    return aspect_ratio, bucket

class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        
    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        if self.start_time is None:
            return 0
        end = self.end_time or time.time()
        return end - self.start_time