"""
Medical Vision Inference System for Llama 3.2 11B Vision Model
Handles medical image analysis and diagnostic text generation with optimized performance.
"""

import os
import sys
import time
import psutil
import inspect
import traceback
import numpy as np
import GPUtil
import torch
from pathlib import Path
from typing import Dict, Optional, Union, Any
from PIL import Image

# Add the project root directory to Python path for imports
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    AutoTokenizer
)

from Models.vision_model.config import SystemConfig
from Models.vision_model.logger import MedicalVisionLogger
from Models.vision_model.utils import setup_gpu_device, validate_image, get_aspect_ratio_info, Timer
from Models.vision_model.errors import (
    ModelLoadError, GPUMemoryError, ImageProcessingError, 
    InferenceError, ValidationError, TimeoutError
)



class MedicalVisionInference:
    """Medical Vision Inference system using Llama 3.2 11B Vision Model."""
    
    def __init__(
        self, 
        model_path: Union[str, Path],
        config: Optional[SystemConfig] = None,
        log_file: Optional[Path] = None
    ):
        """Initialize the Medical Vision Inference system."""
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise ModelLoadError(f"Model path not found: {model_path}")
        
        # Initialize configuration
        self.config = config or SystemConfig()
        
        # Set up logging
        self.logger = MedicalVisionLogger("MedicalVision", log_file)
        
        # Initialize components
        self.model = None
        self.processor = None
        self.tokenizer = None
        
        # Set up device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Clear GPU memory if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
        # Load the model components
        self.load_model()

    def validate_gpu_requirements(self):
        """Validate GPU requirements before loading.
        
        Returns:
            tuple: (bool, str) - (is_valid, error_message)
        """
        if not torch.cuda.is_available():
            return False, "CUDA not available"
            
        # Clear cache before checking memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        gpu = torch.cuda.get_device_properties(0)
        total_mem = gpu.total_memory / (1024**3)
        allocated_mem = torch.cuda.memory_allocated(0) / (1024**3)
        reserved_mem = torch.cuda.memory_reserved(0) / (1024**3)
        
        # Calculate truly available memory
        available_mem = total_mem - reserved_mem
        
        self.logger.info("GPU Memory Analysis:")
        self.logger.info(f"   Device: {gpu.name}")
        self.logger.info(f"   Total: {total_mem:.1f}GB")
        self.logger.info(f"   Reserved: {reserved_mem:.1f}GB")
        self.logger.info(f"   Allocated: {allocated_mem:.1f}GB")
        self.logger.info(f"   Available: {available_mem:.1f}GB")
        
        if available_mem < self.config.memory.min_gpu_memory:
            return False, f"Insufficient GPU memory: {available_mem:.1f}GB (minimum {self.config.memory.min_gpu_memory}GB required)"
            
        return True, None
    
    def load_model(self):
        """Load model with optimizations for 8GB VRAM.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        
        Raises:
            RuntimeError: If GPU requirements are not met
            Exception: For other initialization errors
        """
        self.logger.info("\nLoading model with optimizations...")
        self.logger.info("=" * 50)
        
        try:
            # Validate GPU requirements
            is_valid, error_msg = self.validate_gpu_requirements()
            if not is_valid:
                raise RuntimeError(f"GPU validation failed: {error_msg}")
            
            # Load processor first
            self.logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            
            # Ensure we have a tokenizer
            self.logger.info("Loading tokenizer...")
            try:
                # First try to get tokenizer from processor
                self.tokenizer = getattr(self.processor, "tokenizer", None)
                
                # If not available, load it separately
                if self.tokenizer is None:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                    
                # Verify tokenizer has required attributes
                required_attrs = ['pad_token_id', 'eos_token_id']
                missing_attrs = [attr for attr in required_attrs if not hasattr(self.tokenizer, attr)]
                
                if missing_attrs:
                    raise AttributeError(f"Tokenizer missing required attributes: {missing_attrs}")
                
                self.logger.info("Tokenizer loaded successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to load tokenizer: {str(e)}")
                raise ModelLoadError("Failed to initialize tokenizer with required attributes")
            
            # Setup conservative 4-bit quantization with CPU fallback
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_quant_storage_dtype=torch.float16
            )
            self.logger.info("Initializing model with optimized 4-bit quantization...")
            
            # Calculate available GPU memory
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.logger.info(f"Total GPU Memory: {gpu_mem:.2f}GB")
            
            # Use simpler memory management
            self.logger.info("Configuring memory management...")
            
            # Configure memory and device mapping
            self.logger.info("Configuring automatic device mapping...")
            
            # Use full GPU memory
            device_map = "auto"  # Let transformers handle device mapping
            
            # Calculate optimal memory allocation
            free_mem = torch.cuda.memory_allocated(0) / (1024**3)
            available_mem = gpu_mem - free_mem
            gpu_limit = min(available_mem * 0.9, 7.5)  # Use 90% of available memory, max 7.5GB
            
            max_memory = {
                0: f"{gpu_limit:.1f}GB",  # Dynamic GPU memory limit
                "cpu": "16GB"    # Generous CPU limit for any overflow
            }
            
            self.logger.info("Memory configuration:")
            self.logger.info(f"   GPU: {max_memory[0]}")
            self.logger.info(f"   CPU: {max_memory['cpu']}")
            
            self.logger.info("Using maximum GPU memory strategy")
            
            # Load model with optimizations
            self.logger.info("Loading model...")
            start_time = time.time()
            
            # Load model with optimizations
            self.logger.info("Loading vision-language model...")
            
            # Load model with robust error handling and retries
            max_retries = 3
            retry_count = 0
            last_exception = None
            
            while retry_count < max_retries:
                try:
                    self.logger.info(f"\nLoading attempt {retry_count + 1}/{max_retries}...")
                    
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        self.model_path,
                        device_map=device_map,
                        max_memory=max_memory,
                        quantization_config=quantization_config,
                        low_cpu_mem_usage=True,
                        dtype=torch.float16,
                        trust_remote_code=True
                    )
                    
                    # Verify model loaded correctly
                    if not hasattr(self.model, 'generate'):
                        raise RuntimeError("Loaded model does not have generation capability")

                    # Inspect and log model configuration
                    self.logger.info("\nModel Configuration Analysis:")
                    self.logger.info("=" * 50)
                    
                    try:
                        config = self.model.config
                        # Check main config
                        self.logger.info("Main config attributes:")
                        for attr in ['model_type', 'num_attention_heads', 'hidden_size']:
                            if hasattr(config, attr):
                                self.logger.info(f"   {attr}: {getattr(config, attr)}")
                        
                        # Check vision config
                        if hasattr(config, 'vision_config'):
                            vision_config = config.vision_config
                            self.logger.info("\nVision config details:")
                            for attr in ['image_size', 'patch_size', 'num_patches', 'num_channels', 
                                       'hidden_size', 'num_attention_heads']:
                                if hasattr(vision_config, attr):
                                    self.logger.info(f"   {attr}: {getattr(vision_config, attr)}")
                        
                        # Check processor config
                        if hasattr(self.processor, 'image_processor'):
                            img_proc = self.processor.image_processor
                            self.logger.info("\nImage processor configuration:")
                            for attr in ['size', 'do_resize', 'patch_size']:
                                if hasattr(img_proc, attr):
                                    self.logger.info(f"   {attr}: {getattr(img_proc, attr)}")
                    except Exception as e:
                        self.logger.warning(f"Non-critical error inspecting config: {str(e)}")
                        
                    self.logger.info("=" * 50)
                    self.logger.info("Model loaded successfully!")
                    break
                    
                except Exception as e:
                    last_exception = e
                    retry_count += 1
                    if retry_count < max_retries:
                        self.logger.warning(f"Loading failed, retrying... ({str(e)})")
                        torch.cuda.empty_cache()  # Clear GPU memory before retry
                        time.sleep(2)  # Wait before retry
                    else:
                        self.last_error = str(e)
                        raise RuntimeError(f"Model loading failed after {max_retries} attempts: {str(e)}")

            # Optional: Enable gradient checkpointing for memory efficiency
            if hasattr(self.model, "enable_gradient_checkpointing"):
                self.model.enable_gradient_checkpointing()
            
            # Ensure model is on the correct device
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                # Move model to CUDA if not already there
                if not all(p.device.type == "cuda" for p in self.model.parameters()):
                    self.model.to(self.device)
                    self.logger.info(f"Model moved to device: {self.device}")
                
                # Log memory usage
                self.logger.info(f"Memory allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f}GB")
                self.logger.info(f"Memory reserved: {torch.cuda.memory_reserved(0)/1024**3:.2f}GB")

            load_time = time.time() - start_time
            self.logger.info(f"Model loaded in {load_time:.2f} seconds")

            # Print memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
                self.logger.info(f"GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")

            return True

        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False

    def prepare_inputs_with_validation(self, image, prompt):
        """Prepare model inputs with comprehensive validation using processor defaults.
        
        Args:
            image: PIL Image to process
            prompt: Text prompt for analysis
            
        Returns:
            dict: Validated inputs ready for model
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If processing fails
        """
        try:
            # Initial basic validation
            if not isinstance(image, Image.Image):
                raise ValueError("Input must be a PIL Image")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("Prompt must be a non-empty string")
                
            # Let the processor handle everything with its default settings
            self.logger.info("Processing inputs with validation...")
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            )
            
            # Validate all tensors with detailed checking
            required_tensors = {
                "input_ids": (torch.long, 2),  # (batch_size, seq_length)
                "attention_mask": (torch.long, 2),  # (batch_size, seq_length)
                "pixel_values": (torch.float32, 6)  # (batch_size, seq_len, num_frames, channels, height, width)
            }
            
            for key, (dtype, expected_dims) in required_tensors.items():
                # Check existence
                if key not in inputs:
                    raise ValueError(f"Missing required tensor: {key}")
                    
                tensor = inputs[key]
                
                # Check for None
                if tensor is None:
                    raise ValueError(f"Tensor {key} is None")
                    
                # Check type
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(f"{key} must be a tensor, got {type(tensor)}")
                    
                # Check dimensions
                if len(tensor.shape) != expected_dims:
                    raise ValueError(f"{key} has wrong dimensions: {tensor.shape}, expected {expected_dims}D")
                    
                # Check for invalid values
                if torch.isnan(tensor).any():
                    raise ValueError(f"{key} contains NaN values")
                if torch.isinf(tensor).any():
                    raise ValueError(f"{key} contains infinite values")
                    
                # Convert dtype if needed
                if tensor.dtype != dtype:
                    self.logger.warning(f"Converting {key} from {tensor.dtype} to {dtype}")
                    inputs[key] = tensor.to(dtype)
            
            # Move all inputs to device and ensure they're contiguous
            validated_inputs = {}
            for key, tensor in inputs.items():
                if isinstance(tensor, torch.Tensor):
                    validated_inputs[key] = tensor.clone().to(self.device).contiguous()
                else:
                    validated_inputs[key] = tensor
            
            # Fix malformed tensors that the processor sometimes creates
            if "aspect_ratio_ids" in validated_inputs:
                # Fix aspect_ratio_ids shape - should be (batch_size,) not (batch_size, 1) or empty
                aspect_ratio_ids = validated_inputs["aspect_ratio_ids"]
                if len(aspect_ratio_ids.shape) > 1:
                    validated_inputs["aspect_ratio_ids"] = aspect_ratio_ids.squeeze()
                elif len(aspect_ratio_ids.shape) == 0:  # Empty tensor - fix it
                    # Recreate with proper batch size
                    batch_size = validated_inputs["input_ids"].shape[0]
                    validated_inputs["aspect_ratio_ids"] = torch.zeros((batch_size,), dtype=torch.long, device=self.device)
                    self.logger.info(f"Fixed empty aspect_ratio_ids, new shape: {validated_inputs['aspect_ratio_ids'].shape}")
            
            # Remove problematic tensors with invalid shapes
            keys_to_remove = []
            for key, tensor in validated_inputs.items():
                if isinstance(tensor, torch.Tensor):
                    # Remove tensors with 0 dimensions (invalid)
                    if 0 in tensor.shape:
                        self.logger.warning(f"Removing {key} with invalid shape: {tensor.shape}")
                        keys_to_remove.append(key)
                    # Check for aspect_ratio_mask being None
                    elif key == "aspect_ratio_mask" and tensor is None:
                        self.logger.warning(f"Removing None {key}")
                        keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del validated_inputs[key]
            
            # Log the shapes that the processor created
            self.logger.info("Processor-generated input shapes (after cleanup):")
            for key, tensor in validated_inputs.items():
                if isinstance(tensor, torch.Tensor):
                    self.logger.info(f"  {key}: {tensor.shape}, dtype={tensor.dtype}")
            
            return validated_inputs
        except Exception as e:
            self.logger.error(f"Error preparing inputs: {str(e)}")
            raise

    def analyze_medical_image(
        self,
        image_path: Union[str, Path],
        prompt: Optional[str] = None,
        generation_params: Optional[Dict[str, Any]] = None,
        timeout: int = 300,  # 5 minutes timeout
        max_retries: int = 3  # Maximum retry attempts
    ) -> Dict[str, Any]:
        """Analyze a medical image and generate detailed findings.
        
        Args:
            image_path: Path to medical image file
            prompt: Optional custom analysis prompt
            generation_params: Optional generation parameters override
            timeout: Maximum time in seconds for the operation
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dict containing:
                - analysis: Generated analysis text
                - inference_time: Model inference time
                - total_time: Total processing time
                - prompt: Used prompt
                - success: Operation success flag
                - error: Error message if failed
                
        Raises:
            ValidationError: If image path is invalid
            ModelLoadError: If model is not initialized
            ImageProcessingError: If image processing fails
            InferenceError: If model inference fails
            TimeoutError: If operation exceeds timeout
        """
        # Validate inputs
        process_start = time.time()
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                raise ValidationError(f"Image not found: {image_path}")
            
            if not self.verify_model_state():
                raise ModelLoadError("Model not properly initialized")

            self.logger.info(f"Analyzing medical image: {image_path.name}")
            
            # Load and validate image
            # Load and validate image
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            self.logger.info(f"Image loaded: {image.size} - Mode: {image.mode}")
            
            # Prepare prompt
            if prompt is None:
                prompt = (
                    "Please analyze this medical chest X-ray image. Describe:\n"
                    "1. Type of imaging and quality\n"
                    "2. Key anatomical findings\n"
                    "3. Any abnormalities or pathological features\n"
                    "4. Possible differential diagnoses\n"
                    "5. Recommended next steps or further investigations\n"
                    "Provide a concise, professional medical assessment."
                )
            
            # Prepare inputs with validation
            inputs = self.prepare_inputs_with_validation(image, prompt)
            
            # Use the inputs exactly as the processor created them, with minimal modifications
            generation_inputs = inputs.copy()
            
            # Use the inputs exactly as the processor created them
            generation_inputs = {}
            
            # Copy and move tensors to the correct device
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    generation_inputs[key] = value.to(self.device)
                else:
                    generation_inputs[key] = value
            
            # Create or fix aspect_ratio_ids if needed
            if "aspect_ratio_ids" not in generation_inputs:
                aspect_ratio = image.size[0] / image.size[1]
                aspect_ratio_id = 1 if aspect_ratio < 0.75 else 2 if aspect_ratio > 1.33 else 0
                generation_inputs["aspect_ratio_ids"] = torch.full(
                    (inputs["input_ids"].shape[0],), aspect_ratio_id, 
                    dtype=torch.long, device=self.device
                )
            
            # Create aspect_ratio_mask if it's missing or None
            if "aspect_ratio_mask" not in generation_inputs or generation_inputs["aspect_ratio_mask"] is None:
                batch_size = inputs["input_ids"].shape[0]
                generation_inputs["aspect_ratio_mask"] = torch.ones(
                    (batch_size, 4), dtype=torch.bool, device=self.device
                )
                self.logger.info(f"Created aspect_ratio_mask with shape: {generation_inputs['aspect_ratio_mask'].shape}")

            # Log final generation inputs
            self.logger.info("Final generation inputs:")
            for key, value in generation_inputs.items():
                if isinstance(value, torch.Tensor):
                    self.logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
            
            # Generation parameters - optimized to reduce repetition
            gen_params = {
                "max_new_tokens": 200,
                "do_sample": True,          # Enable sampling to reduce repetition
                "temperature": 0.7,         # Moderate randomness
                "top_p": 0.9,              # Nucleus sampling
                "repetition_penalty": 1.2,  # Penalize repetition
                "no_repeat_ngram_size": 3,  # Prevent 3-gram repetition
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id
            }
            
            # Add any custom generation parameters
            if generation_params:
                gen_params.update(generation_params)
            
            # Perform generation with detailed error logging
            self.logger.info("Starting model inference...")
            gen_start = time.time()
            
            try:
                # Add detailed debugging before generation
                self.logger.info("Pre-generation validation:")
                for k, v in generation_inputs.items():
                    if v is None:
                        raise ValueError(f"Input {k} is None before generation")
                    elif isinstance(v, torch.Tensor):
                        self.logger.info(f"  {k}: shape={v.shape}, dtype={v.dtype}, device={v.device}, requires_grad={v.requires_grad}")
                        if torch.isnan(v).any():
                            raise ValueError(f"Input {k} contains NaN values")
                        if torch.isinf(v).any():
                            raise ValueError(f"Input {k} contains infinite values")
                
                # Clear any gradients and ensure no grad computation
                with torch.no_grad():
                    # Use all inputs including the fixed aspect_ratio_mask
                    self.logger.info("Attempting generation with all required inputs...")
                    outputs = self.model.generate(
                        **generation_inputs,
                        **gen_params
                    )
                
                if outputs is None or len(outputs) == 0:
                    raise RuntimeError("Model generated empty output")
                
                # Decode the output
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Remove the prompt from the generated text
                if prompt in generated_text:
                    analysis = generated_text.replace(prompt, "").strip()
                else:
                    analysis = generated_text.strip()
                
                # Clean up repetitive text (simple approach)
                sentences = analysis.split('. ')
                cleaned_sentences = []
                seen_sentences = set()
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence and sentence not in seen_sentences:
                        cleaned_sentences.append(sentence)
                        seen_sentences.add(sentence)
                    elif len(cleaned_sentences) >= 3:  # Stop after 3 unique sentences if repetition starts
                        break
                
                analysis = '. '.join(cleaned_sentences)
                if analysis and not analysis.endswith('.'):
                    analysis += '.'
                
                gen_time = time.time() - gen_start
                total_time = time.time() - process_start
                
                self.logger.info(f"Analysis completed successfully in {gen_time:.2f}s")
                self.logger.info(f"Generated text length: {len(analysis)} characters")
                
                return {
                    'analysis': analysis,
                    'inference_time': gen_time,
                    'total_time': total_time,
                    'prompt': prompt,
                    'success': True
                }
                
            except Exception as e:
                # Log detailed error information
                self.logger.error(f"Generation failed with error: {str(e)}")
                self.logger.error(f"Error type: {type(e).__name__}")
                
                # Try to get more specific error info
                import traceback
                self.logger.error("Full traceback:")
                self.logger.error(traceback.format_exc())
                
                raise InferenceError(f"Model inference failed: {str(e)}")
            
        except Exception as e:
            total_time = time.time() - process_start
            self.logger.error(f"Analysis failed after {total_time:.2f}s: {str(e)}")
            return {
                'analysis': None,
                'inference_time': 0,
                'total_time': total_time,
                'prompt': prompt,
                'error': str(e),
                'success': False
            }
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def verify_model_state(self) -> bool:
        """Verify model is in a valid state for inference.
        
        Returns:
            bool: True if model is ready for inference, False otherwise
        """

    def verify_model_state(self) -> bool:
        """Verify model is in a valid state for inference.
        
        Returns:
            bool: True if model is ready for inference, False otherwise
        """
        try:
            # Check basic initialization
            if self.model is None or self.processor is None or self.tokenizer is None:
                self.logger.error("Model components not initialized")
                return False
            
            if not hasattr(self.model, 'generate'):
                self.logger.error("Model missing generate method")
                return False
            
            # Verify device placement
            model_devices = {p.device for p in self.model.parameters()}
            if len(model_devices) > 1:
                self.logger.error(f"Model parameters spread across multiple devices: {model_devices}")
                return False
                
            model_device = next(self.model.parameters()).device
            if model_device != self.device:
                self.logger.error(f"Model on device {model_device}, expected {self.device}")
                # Try to fix the device placement
                try:
                    self.model.to(self.device)
                    self.logger.info(f"Successfully moved model to {self.device}")
                except Exception as e:
                    self.logger.error(f"Failed to move model to {self.device}: {e}")
                    return False
                
            # Verify CUDA availability if needed
            if self.device.type == "cuda" and not torch.cuda.is_available():
                self.logger.error("CUDA device requested but not available")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error verifying model state: {e}")
            self.logger.error(traceback.format_exc())
            return False

    def get_system_info(self, detailed: bool = False):
        """Get comprehensive system resource usage information.
        
        Args:
            detailed: Whether to include detailed GPU memory analysis
            
        Returns:
            dict: System resource information including:
                - CPU usage (total and per core)
                - RAM usage (used, available, total)
                - GPU information (if available):
                    - Name and architecture
                    - Memory usage (used, free, total)
                    - Temperature
                    - Power usage
                    - Compute mode
        """
        info = {
            'timestamp': time.time(),
            'cpu': {
                'total_percent': psutil.cpu_percent(),
                'per_cpu_percent': psutil.cpu_percent(percpu=True),
                'count': psutil.cpu_count(),
                'count_logical': psutil.cpu_count(logical=True)
            },
            'memory': {
                'total': psutil.virtual_memory().total / (1024**3),
                'available': psutil.virtual_memory().available / (1024**3),
                'used': psutil.virtual_memory().used / (1024**3),
                'percent': psutil.virtual_memory().percent
            },
            'swap': {
                'total': psutil.swap_memory().total / (1024**3),
                'used': psutil.swap_memory().used / (1024**3),
                'free': psutil.swap_memory().free / (1024**3),
                'percent': psutil.swap_memory().percent
            }
        }
        
        if torch.cuda.is_available():
            try:
                gpu = GPUtil.getGPUs()[0]
                gpu_info = {
                    'name': gpu.name,
                    'load_percent': gpu.load * 100,
                    'memory': {
                        'total': gpu.memoryTotal,
                        'used': gpu.memoryUsed,
                        'free': gpu.memoryFree,
                        'percent': (gpu.memoryUsed / gpu.memoryTotal) * 100
                    },
                    'temperature': gpu.temperature,
                    'uuid': gpu.uuid
                }
                
                if detailed:
                    # Add CUDA-specific info
                    gpu_info.update({
                        'cuda': {
                            'allocated': torch.cuda.memory_allocated(0) / (1024**3),
                            'reserved': torch.cuda.memory_reserved(0) / (1024**3),
                            'max_allocated': torch.cuda.max_memory_allocated(0) / (1024**3),
                            'max_reserved': torch.cuda.max_memory_reserved(0) / (1024**3)
                        }
                    })
                    
                    # Get device properties
                    props = torch.cuda.get_device_properties(0)
                    gpu_info['device_properties'] = {
                        'compute_capability': f"{props.major}.{props.minor}",
                        'total_memory': props.total_memory / (1024**2),
                        'multi_processor_count': props.multi_processor_count,
                        'max_threads_per_block': props.max_threads_per_block,
                        'max_threads_per_multiprocessor': props.max_threads_per_multi_processor,
                        'warp_size': props.warp_size
                    }
                
                info['gpu'] = gpu_info
                
            except Exception as e:
                self.logger.warning(f"Error getting GPU info: {str(e)}")
                info['gpu'] = {'error': str(e)}
        
        return info


def main():
    """Test medical image analysis with the vision model."""
    logger = MedicalVisionLogger("TestRunner")
    logger.info("=" * 70)
    logger.info("🏥 MEDICAL VISION INFERENCE TEST")
    logger.info("=" * 70)
    
    try:
        # Initialize with configuration
        config = SystemConfig()
        model_path = Path(__file__).parent / "llama-3.2-11b-vision-local"
        inference = MedicalVisionInference(
            model_path=model_path,
            config=config,
            log_file=Path("medical_vision.log")
        )
        
        # Load model
        if not inference.load_model():
            logger.error("Failed to load model")
            return

        # Find test images
        test_images_dir = os.path.join(os.path.dirname(__file__), "test_images")
        if not os.path.exists(test_images_dir):
            os.makedirs(test_images_dir)
            logger.info(f"Created test images directory: {test_images_dir}")
            return
        
        image_files = [f for f in os.listdir(test_images_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.dicom', '.dcm'))]
    
        if not image_files:
            logger.warning(f"No medical images found in: {test_images_dir}")
            return
        
        # Analyze first image found
        image_path = os.path.join(test_images_dir, image_files[0])
        
        # Get system status before analysis
        logger.info("System Status Before Analysis:")
        system_info = inference.get_system_info()
        for key, value in system_info.items():
            logger.info(f"{key}: {value}")
        
        # Analyze image
        result = inference.analyze_medical_image(image_path)
        
        if result and result.get('success', False):
            logger.info("Analysis completed successfully")
            logger.info("Generated Analysis:")
            logger.info("-" * 50)
            logger.info(result['analysis'])
            logger.info("-" * 50)
        else:
            logger.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
        
        # Get system status after analysis
        logger.info("System Status After Analysis:")
        system_info = inference.get_system_info()
        for key, value in system_info.items():
            logger.info(f"{key}: {value}")
            
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")

if __name__ == "__main__":
    main()