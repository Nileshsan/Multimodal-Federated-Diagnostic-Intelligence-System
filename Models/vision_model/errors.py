"""Custom exceptions for Medical Vision Inference system."""

class MedicalVisionError(Exception):
    """Base exception for Medical Vision Inference errors."""
    pass

class ModelLoadError(MedicalVisionError):
    """Raised when model loading fails."""
    pass

class GPUMemoryError(MedicalVisionError):
    """Raised when GPU memory requirements are not met."""
    pass

class ImageProcessingError(MedicalVisionError):
    """Raised when image processing fails."""
    pass

class InferenceError(MedicalVisionError):
    """Raised when inference fails."""
    pass

class TokenizerError(MedicalVisionError):
    """Raised when tokenizer operations fail."""
    pass

class ValidationError(MedicalVisionError):
    """Raised when input validation fails."""
    pass

class TimeoutError(MedicalVisionError):
    """Raised when operation times out."""
    pass