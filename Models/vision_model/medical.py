"""
Fixed Medical Vision Inference System for Llama 3.2 11B Vision Model
Addresses the 'NoneType' object has no attribute 'reshape' error.
"""

import os
import sys
import time
import psutil
import inspect
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
        self.device = self.config.model.device
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    
    def validate_gpu_requirements(self):
        """Validate GPU requirements before loading."""
        if not torch.cuda.is_available():
            return False, "CUDA not available"
            
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        gpu = torch.cuda.get_device_properties(0)
        total_mem = gpu.total_memory / (1024**3)
        allocated_mem = torch.cuda.memory_allocated(0) / (1024**3)
        reserved_mem = torch.cuda.memory_reserved(0) / (1024**3)
        available_mem = total_mem - reserved_mem
        
        self.logger.info("GPU Memory Analysis:")
        self.logger.info(f"   Device: {gpu.name}")
        self.logger.info(f"   Total: {total_mem:.1f}GB")
        self.logger.info(f"   Available: {available_mem:.1f}GB")
        
        if available_mem < self.config.memory.min_gpu_memory:
            return False, f"Insufficient GPU memory: {available_mem:.1f}GB"
            
        return True, None
    
    def load_model(self):
        """Load model with optimizations for 8GB VRAM."""
        self.logger.info("\nLoading model with optimizations...")
        
        try:
            # Validate GPU requirements
            is_valid, error_msg = self.validate_gpu_requirements()
            if not is_valid:
                raise RuntimeError(f"GPU validation failed: {error_msg}")
            
            start_time = time.time()
            
            # Load processor and tokenizer
            self.logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            
            self.logger.info("Loading tokenizer...")
            try:
                self.tokenizer = getattr(self.processor, "tokenizer", None)
                if self.tokenizer is None:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.logger.info("Tokenizer loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load tokenizer: {str(e)}")
                raise ModelLoadError("Failed to initialize tokenizer")
            
            # Setup quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_quant_storage_dtype=torch.float16
            )
            
            # Load model with optimizations
            self.logger.info("Loading vision-language model...")
            
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                device_map="auto",
                max_memory={0: "7.2GB", "cpu": "16GB"},
                quantization_config=quantization_config,
                low_cpu_mem_usage=True,
                dtype=torch.float16,
                trust_remote_code=True
            )
            
            load_time = time.time() - start_time
            self.logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def prepare_inputs_with_validation(self, image, prompt):
        """Prepare model inputs with comprehensive validation using processor defaults."""
        try:
            # Let the processor handle everything with its default settings
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            )
            
            # Validate all required inputs are present and not None
            required_keys = ["input_ids", "attention_mask", "pixel_values"]
            for key in required_keys:
                if key not in inputs:
                    raise ValueError(f"Missing required input: {key}")
                if inputs[key] is None:
                    raise ValueError(f"Input {key} is None")
            
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
        generation_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze a medical image and generate detailed findings."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise ValidationError(f"Image not found: {image_path}")
        
        if not hasattr(self.model, 'generate'):
            raise ModelLoadError("Model not properly initialized")

        self.logger.info(f"Analyzing medical image: {image_path.name}")
        
        process_start = time.time()
        try:
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
            
            # Create or fix aspect_ratio_ids if needed
            if "aspect_ratio_ids" not in generation_inputs:
                aspect_ratio = image.size[0] / image.size[1]
                aspect_ratio_id = 1 if aspect_ratio < 0.75 else 2 if aspect_ratio > 1.33 else 0
                generation_inputs["aspect_ratio_ids"] = torch.full(
                    (inputs["input_ids"].shape[0],), aspect_ratio_id, 
                    dtype=torch.long, device=self.device
                )
            
            # Create aspect_ratio_mask if it's missing or None (this is what caused the error)
            if "aspect_ratio_mask" not in generation_inputs or generation_inputs["aspect_ratio_mask"] is None:
                # Create a proper aspect_ratio_mask
                batch_size = inputs["input_ids"].shape[0]
                # For Llama Vision, aspect_ratio_mask should be (batch_size, num_tiles)
                # Since we have 4 frames/tiles, use shape (batch_size, 4)
                generation_inputs["aspect_ratio_mask"] = torch.ones(
                    (batch_size, 4), dtype=torch.bool, device=self.device
                )
                self.logger.info(f"Created aspect_ratio_mask with shape: {generation_inputs['aspect_ratio_mask'].shape}")
            
            # Log all input shapes for debugging
            self.logger.info("Final generation inputs:")
            for k, v in generation_inputs.items():
                if isinstance(v, torch.Tensor):
                    self.logger.info(f"  {k}: {v.shape}, dtype={v.dtype}, device={v.device}")
            
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
    
    def get_system_info(self):
        """Get current system resource usage."""
        info = {
            'cpu_percent': psutil.cpu_percent(),
            'ram_usage': psutil.virtual_memory().percent
        }
        
        if torch.cuda.is_available():
            gpu = GPUtil.getGPUs()[0]
            info.update({
                'gpu_name': gpu.name,
                'gpu_load': gpu.load * 100,
                'gpu_memory_used': gpu.memoryUsed,
                'gpu_memory_total': gpu.memoryTotal
            })
        
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