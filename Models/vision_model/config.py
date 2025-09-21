"""Configuration management for Medical Vision Inference system."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Union

@dataclass
class ModelConfig:
    """Configuration for the vision model."""
    image_size: Dict[str, int] = field(default_factory=lambda: {"height": 224, "width": 224})
    num_channels: int = 3
    max_sequence_length: int = 512
    dtype: str = "float16"
    device: str = "cuda"

@dataclass
class InferenceConfig:
    """Configuration for inference parameters."""
    max_retries: int = 3
    timeout: int = 300
    generation_params: Dict[str, Union[int, float, bool]] = field(default_factory=lambda: {
        "max_new_tokens": 500,
        "min_new_tokens": 100,
        "temperature": 0.7,
        "do_sample": True,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
        "num_beams": 4,
        "length_penalty": 1.0
    })
    

@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    min_gpu_memory: float = 6.0  # Minimum required GPU memory in GB
    max_gpu_usage: float = 7.5   # Maximum GPU memory usage in GB
    gpu_buffer: float = 0.9      # Buffer for GPU memory (90%)
    cpu_memory: str = "16GB"     # CPU memory allocation

@dataclass
class SystemConfig:
    """Main system configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    quantization: Dict[str, Union[bool, str]] = field(default_factory=lambda: {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4"
    })