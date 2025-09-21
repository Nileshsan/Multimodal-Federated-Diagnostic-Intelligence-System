"""Configuration management for Medical Vision Inference system."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Union

@dataclass
class ModelConfig:
    """Configuration for the vision model."""
    image_size: Dict[str, int] = field(default_factory=lambda: {"height": 560, "width": 560})
    num_channels: int = 3
    max_sequence_length: int = 256  # Reduced for faster inference
    dtype: str = "float16"
    device: str = "cuda"
    enable_torch_compile: bool = True
    torch_compile_mode: str = "max-autotune"
    use_cache: bool = True
    offload_folder: str = "Models/vision_model/offload"

@dataclass
class InferenceConfig:
    """Configuration for inference parameters."""
    max_retries: int = 2  # Reduced retries
    timeout: int = 180    # Reduced timeout
    batch_size: int = 1   # Can be increased if memory allows
    use_cache: bool = True
    enable_memory_optimization: bool = True
    generation_params: Dict[str, Union[int, float, bool]] = field(default_factory=lambda: {
        "max_new_tokens": 256,     # Reduced for faster generation
        "min_new_tokens": 50,      # Reduced minimum tokens
        "temperature": 0.7,
        "do_sample": True,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,  # Slightly reduced
        "no_repeat_ngram_size": 3,
        "num_beams": 2,            # Reduced beam search
        "length_penalty": 1.0,
        "early_stopping": True,     # Enable early stopping
        "use_cache": True          # Enable KV cache
    })
    

@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    min_gpu_memory: float = 4.0  # Reduced minimum required GPU memory
    max_gpu_usage: float = 8.0   # Increased maximum GPU usage
    gpu_buffer: float = 0.95     # Increased buffer utilization (95%)
    cpu_memory: str = "16GB"     # CPU memory allocation
    enable_cpu_offload: bool = True  # Enable CPU offloading
    low_cpu_mem_usage: bool = True   # Enable low CPU memory usage
    use_cache: bool = True           # Enable KV cache
    preload_modules: bool = True     # Preload frequently used modules

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
        "bnb_4bit_quant_type": "nf4",
        "llm_int8_enable_fp32_cpu_offload": True,
        "llm_int8_skip_modules": None,
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4"
    })