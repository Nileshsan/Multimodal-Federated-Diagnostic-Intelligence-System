"""
Medical Vision Inference System for Llama 3.2 11B Vision Model
Fixed version based on working implementation.
"""

import os
import sys
import time
import torch
import psutil
import GPUtil
import traceback
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
from Models.vision_model.errors import ModelLoadError, ValidationError
from Models.vision_model.logger import MedicalVisionLogger
from Models.processors.report_generator import StructuredReportGenerator


class MedicalVisionInference:
    """Medical Vision Inference system using Llama 3.2 11B Vision Model."""
    
    def __init__(
        self, 
        model_path: Union[str, Path],
        config: Optional[SystemConfig] = None,
        log_file: Optional[Path] = None,
        knowledge_base_path: Optional[Path] = None
    ):
        """Initialize the Medical Vision Inference system."""
        # Clear any existing GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize report generator
        if knowledge_base_path:
            self.report_generator = StructuredReportGenerator(
                medical_knowledge_base=knowledge_base_path
            )
            self.logger.info("Initialized structured report generator")
        else:
            self.report_generator = None

        # Configure CUDA settings for better memory management and shared memory access
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Set environment variables for optimal memory allocation - FROM WORKING CODE
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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
            
            # Setup quantization - EXACTLY as in working code
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_quant_storage_dtype=torch.float16
            )
            
            # Load model with optimizations - EXACTLY as in working code with fallback
            self.logger.info("Loading vision-language model...")
            
            try:
                # First attempt with standard configuration
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_path,
                    device_map="auto",
                    max_memory={0: "7.2GB", "cpu": "16GB"},
                    quantization_config=quantization_config,
                    low_cpu_mem_usage=True,
                    dtype=torch.float16,
                    trust_remote_code=True
                )
                self.logger.info("Model loaded successfully on first attempt!")
                
            except torch.cuda.OutOfMemoryError as oom_error:
                self.logger.error(f"First attempt failed with CUDA OOM: {oom_error}")
                
                # Clear all GPU memory
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                torch.cuda.reset_peak_memory_stats()
                
                self.logger.info("Retrying with reduced memory limits...")
                
                # Second attempt with more conservative settings
                try:
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        self.model_path,
                        device_map="auto",
                        max_memory={0: "6.5GB", "cpu": "20GB"},  # Reduce GPU, increase CPU
                        quantization_config=quantization_config,
                        low_cpu_mem_usage=True,
                        dtype=torch.float16,
                        trust_remote_code=True
                    )
                    self.logger.info("Model loaded successfully on second attempt!")
                    
                except Exception as e2:
                    self.logger.error(f"Second attempt also failed: {e2}")
                    
                    # Third attempt with even more aggressive CPU offloading
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    self.logger.info("Final attempt with maximum CPU offloading...")
                    
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        self.model_path,
                        device_map="auto",
                        max_memory={0: "6.0GB", "cpu": "24GB"},  # Even more conservative
                        quantization_config=quantization_config,
                        low_cpu_mem_usage=True,
                        dtype=torch.float16,
                        trust_remote_code=True
                    )
                    self.logger.info("Model loaded successfully on final attempt!")
            
            # Verify model loaded correctly
            if self.model is None:
                raise RuntimeError("Model is None after loading attempts")
            
            load_time = time.time() - start_time
            self.logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False

    def prepare_inputs_with_validation(self, image, prompt):
        """Prepare model inputs with comprehensive validation - COPIED FROM WORKING CODE."""
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

    def generate_structured_report(
        self,
        model_output: str,
        image_analysis: Dict,
        patient_context: Optional[Dict] = None,
        output_format: str = 'json'
    ) -> Union[Dict, str]:
        """Generate a structured medical report."""
        if not self.report_generator:
            return {"error": "No report generator available"}
            
        self.logger.info("Generating structured medical report...")
        
        try:
            report = self.report_generator.generate_structured_report(
                model_output=model_output,
                image_analysis=image_analysis,
                patient_context=patient_context
            )
            
            if output_format.lower() == 'json':
                return self._format_report_as_json(report)
            elif output_format.lower() == 'text':
                return self._format_report_as_text(report)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
                
        except Exception as e:
            self.logger.error(f"Error generating structured report: {str(e)}")
            raise

    def _format_report_as_json(self, report) -> Dict:
        """Format report as JSON."""
        return {
            'clinical_findings': getattr(report, 'clinical_findings', {}),
            'diagnostic_interpretation': getattr(report, 'diagnostic_interpretation', {}),
            'technical_details': getattr(report, 'technical_details', {}),
            'patient_explanation': getattr(report, 'patient_explanation', {}),
            'additional_notes': getattr(report, 'additional_notes', {}),
            'metadata': getattr(report, 'metadata', {})
        }
        
    def _format_report_as_text(self, report) -> str:
        """Format report as text."""
        sections = []
        sections.extend([
            "MEDICAL DIAGNOSTIC REPORT",
            "=" * 30,
            f"Report ID: {getattr(report, 'metadata', {}).get('report_id', 'N/A')}",
            f"Date: {getattr(report, 'metadata', {}).get('timestamp', 'N/A')}\n",
            "CLINICAL FINDINGS",
            "-" * 20
        ])
        
        return "\n".join(sections)

    def analyze_image(
        self,
        image: Union[str, Path, Image.Image],
        prompt: Optional[str] = None,
        generation_params: Optional[Dict] = None
    ) -> Dict:
        """Analyze a medical image and generate a report - MAIN ANALYSIS METHOD."""
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise ValidationError(f"Image not found: {image_path}")
            image = Image.open(image_path)
        
        if not hasattr(self.model, 'generate') or self.model is None:
            raise ModelLoadError("Model not properly initialized")

        self.logger.info(f"Analyzing medical image")
        
        process_start = time.time()
        try:
            # Load and validate image
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
            
            # Prepare inputs with validation - USING WORKING CODE METHOD
            inputs = self.prepare_inputs_with_validation(image, prompt)
            
            # Use the inputs exactly as the processor created them, with minimal modifications
            generation_inputs = inputs.copy()
            
            # Create or fix aspect_ratio_ids if needed - FROM WORKING CODE
            if "aspect_ratio_ids" not in generation_inputs:
                aspect_ratio = image.size[0] / image.size[1]
                aspect_ratio_id = 1 if aspect_ratio < 0.75 else 2 if aspect_ratio > 1.33 else 0
                generation_inputs["aspect_ratio_ids"] = torch.full(
                    (inputs["input_ids"].shape[0],), aspect_ratio_id, 
                    dtype=torch.long, device=self.device
                )
            
            # Create aspect_ratio_mask if it's missing or None - CRITICAL FIX FROM WORKING CODE
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
            
            # Generation parameters - FROM WORKING CODE
            gen_params = {
                "max_new_tokens": 200,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.2,
                "no_repeat_ngram_size": 3,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id
            }
            
            if generation_params:
                gen_params.update(generation_params)
            
            # Perform generation - FROM WORKING CODE
            self.logger.info("Starting model inference...")
            gen_start = time.time()
            
            try:
                # Clear any gradients and ensure no grad computation
                with torch.no_grad():
                    self.logger.info("Attempting generation with all required inputs...")
                    outputs = self.model.generate(
                        **generation_inputs,
                        **gen_params
                    )
                
                if outputs is None or len(outputs) == 0:
                    raise RuntimeError("Model generated empty output")
                
                # Decode the output - FROM WORKING CODE
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Remove the prompt from the generated text
                if prompt in generated_text:
                    analysis = generated_text.replace(prompt, "").strip()
                else:
                    analysis = generated_text.strip()
                
                # Clean up repetitive text - FROM WORKING CODE
                sentences = analysis.split('. ')
                cleaned_sentences = []
                seen_sentences = set()
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence and sentence not in seen_sentences:
                        cleaned_sentences.append(sentence)
                        seen_sentences.add(sentence)
                    elif len(cleaned_sentences) >= 3:
                        break
                
                analysis = '. '.join(cleaned_sentences)
                if analysis and not analysis.endswith('.'):
                    analysis += '.'
                
                gen_time = time.time() - gen_start
                total_time = time.time() - process_start
                
                self.logger.info(f"Analysis completed successfully in {gen_time:.2f}s")
                
                # Try to generate structured report if available
                try:
                    if self.report_generator:
                        image_analysis_data = {
                            'image_type': 'medical',
                            'image_quality': {
                                'dimensions': image.size,
                                'mode': image.mode,
                                'aspect_ratio': image.size[0] / image.size[1]
                            },
                            'processing_time': gen_time
                        }
                        
                        report = self.generate_structured_report(
                            model_output=analysis,
                            image_analysis=image_analysis_data
                        )
                        
                        return {
                            'analysis': analysis,
                            'structured_report': report,
                            'inference_time': gen_time,
                            'total_time': total_time,
                            'prompt': prompt,
                            'success': True
                        }
                except Exception as report_error:
                    self.logger.error(f"Failed to generate structured report: {str(report_error)}")
                
                return {
                    'analysis': analysis,
                    'inference_time': gen_time,
                    'total_time': total_time,
                    'prompt': prompt,
                    'success': True
                }
                
            except Exception as e:
                self.logger.error(f"Generation failed with error: {str(e)}")
                self.logger.error("Full traceback:")
                self.logger.error(traceback.format_exc())
                raise
            
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

    def analyze_medical_image(
        self,
        image_path: Union[str, Path],
        diagnostic_type: str = 'general',
        custom_prompt: Optional[str] = None
    ) -> Dict:
        """Analyze a medical image (compatibility method)."""
        prompt = custom_prompt
        if prompt is None:
            if diagnostic_type == 'general':
                prompt = (
                    "Please analyze this medical image. Describe:\n"
                    "1. Type of imaging and quality\n"
                    "2. Key anatomical findings\n"
                    "3. Any abnormalities or pathological features\n"
                    "4. Possible differential diagnoses\n"
                    "5. Recommended next steps\n"
                    "Provide a professional medical assessment."
                )
            elif diagnostic_type == 'detailed':
                prompt = (
                    "Please provide a detailed analysis of this medical image:\n"
                    "1. Image type, quality, and positioning\n"
                    "2. Comprehensive anatomical description\n"
                    "3. Detailed pathological findings\n"
                    "4. Measurements and comparisons\n"
                    "5. Differential diagnoses with likelihood\n"
                    "6. Recommended follow-up studies\n"
                    "7. Clinical correlation suggestions"
                )
            else:
                prompt = (
                    "Analyze this medical image focusing on:\n"
                    f"- Specific diagnostic type: {diagnostic_type}\n"
                    "- Key findings and abnormalities\n"
                    "- Relevant measurements\n"
                    "- Differential diagnoses\n"
                    "- Recommendations"
                )
        
        return self.analyze_image(image_path, prompt=prompt)

    def get_system_info(self, detailed: bool = False):
        """Get current system resource usage."""
        info = {
            'timestamp': time.time(),
            'cpu': {
                'total_percent': psutil.cpu_percent(),
                'count': psutil.cpu_count()
            },
            'memory': {
                'total': psutil.virtual_memory().total / (1024**3),
                'available': psutil.virtual_memory().available / (1024**3),
                'used': psutil.virtual_memory().used / (1024**3),
                'percent': psutil.virtual_memory().percent
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
            
            # Log performance metrics
            logger.info(f"Inference time: {result.get('inference_time', 0):.2f} seconds")
            logger.info(f"Total time: {result.get('total_time', 0):.2f} seconds")
            
            # Show structured report if available
            if 'structured_report' in result:
                logger.info("Structured report generated successfully")
        else:
            logger.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
        
        # Get system status after analysis
        logger.info("System Status After Analysis:")
        system_info = inference.get_system_info(detailed=True)
        for key, value in system_info.items():
            logger.info(f"{key}: {value}")
            
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()