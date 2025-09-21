"""
Medical Vision Inference System for Llama 3.2 11B Vision Model
Handles medical image analysis and diagnostic text generation with optimized performance.
"""

import os
import sys
import time
import torch
import psutil
import GPUtil
import inspect
import numpy as np
import traceback
from pathlib import Path
from typing import Dict, Optional, Union, Any, List
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
        
        # Initialize report generator
        self.report_generator = StructuredReportGenerator(
            medical_knowledge_base=knowledge_base_path
        )
        self.logger.info("Initialized structured report generator")
        
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

    def load_model(self) -> bool:
        """Load the model and its components."""
        try:
            self.logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            
            self.logger.info("Loading tokenizer...")
            self.tokenizer = getattr(self.processor, "tokenizer", None)
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_quant_storage_dtype=torch.float16
            )
            
            # Configure GPU memory management
            if torch.cuda.is_available():
                # Analyze GPU memory
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                free_mem = torch.cuda.memory_allocated(0) / (1024**3)
                available_mem = gpu_mem - free_mem
                
                self.logger.info("\nGPU Memory Analysis:")
                self.logger.info("   Device: " + torch.cuda.get_device_name(0))
                self.logger.info(f"   Total: {gpu_mem:.1f}GB")
                self.logger.info(f"   Reserved: {free_mem:.1f}GB")
                self.logger.info(f"   Allocated: {torch.cuda.memory_allocated(0)/(1024**3):.1f}GB")
                self.logger.info(f"   Available: {available_mem:.1f}GB")
                
                # Configure memory more aggressively
                gpu_limit = min(available_mem * 0.95, available_mem - 0.5)  # Use 95% of available memory, keep 0.5GB buffer
                
                # Set up memory configuration
                max_memory = {
                    0: f"{gpu_limit:.1f}GB",  # GPU memory
                    "cpu": "16GB"  # CPU memory
                }
                
                self.logger.info("Configuring memory management...")
                self.logger.info("Configuring automatic device mapping...")
                self.logger.info("Memory configuration:")
                self.logger.info(f"   GPU: {gpu_limit:.1f}GB")
                self.logger.info(f"   CPU: 16GB")
                self.logger.info("Using maximum GPU memory strategy")
            else:
                max_memory = None
                
            # Configure model loading with optimizations
            self.logger.info("Loading model with optimizations...")
            
            # Set up torch compile for faster inference
            if torch.cuda.is_available():
                torch._dynamo.config.suppress_errors = True
                torch._dynamo.config.cache_size_limit = 64
            
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                device_map="auto",
                max_memory=max_memory,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                offload_folder="Models/vision_model/offload"
            )
            
            # Enable model optimizations
            if torch.cuda.is_available():
                self.model.to(self.device)
                if hasattr(self.model, "config"):
                    self.model.config.use_cache = True
                self.model = torch.compile(
                    self.model,
                    mode="max-autotune",
                    fullgraph=True,
                    dynamic=True
                )
            
            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise ModelLoadError(f"Failed to load model: {str(e)}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def prepare_inputs_with_validation(self, image: Image.Image, prompt: str) -> Dict:
        """Prepare model inputs with validation and optimization."""
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL Image")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string")
            
        self.logger.info("Processing inputs with validation...")
        try:
            # Pre-process image for optimization
            image = image.convert('RGB')
            if max(image.size) > 560:  # Max dimension optimization
                image.thumbnail((560, 560), Image.Resampling.LANCZOS)
            
            # Use torch.jit for processor if available
            if hasattr(self.processor, "torch_jit"):
                processor_fn = self.processor.torch_jit
            else:
                processor_fn = self.processor
                
            # Process inputs with optimized settings
            inputs = processor_fn(
                images=image,
                text=prompt,
                return_tensors="pt",
                padding="max_length",
                max_length=512,
                truncation=True
            )
            
            validated_inputs = {}
            for key, tensor in inputs.items():
                if isinstance(tensor, torch.Tensor):
                    validated_inputs[key] = tensor.clone().to(self.device).contiguous()
                else:
                    validated_inputs[key] = tensor
            
            if "aspect_ratio_ids" in validated_inputs:
                aspect_ratio_ids = validated_inputs["aspect_ratio_ids"]
                if len(aspect_ratio_ids.shape) > 1:
                    validated_inputs["aspect_ratio_ids"] = aspect_ratio_ids.squeeze()
                elif len(aspect_ratio_ids.shape) == 0:
                    batch_size = validated_inputs["input_ids"].shape[0]
                    validated_inputs["aspect_ratio_ids"] = torch.zeros(
                        (batch_size,),
                        dtype=torch.long,
                        device=self.device
                    )
            
            keys_to_remove = []
            for key, tensor in validated_inputs.items():
                if isinstance(tensor, torch.Tensor) and 0 in tensor.shape:
                    self.logger.warning(f"Removing {key} with invalid shape: {tensor.shape}")
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del validated_inputs[key]
            
            return validated_inputs
        except Exception as e:
            raise ValidationError(f"Input validation failed: {str(e)}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def generate_structured_report(
        self,
        model_output: str,
        image_analysis: Dict,
        patient_context: Optional[Dict] = None,
        output_format: str = 'json'
    ) -> Union[Dict, str]:
        """Generate a structured medical report."""
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
            'clinical_findings': report.clinical_findings,
            'diagnostic_interpretation': report.diagnostic_interpretation,
            'technical_details': report.technical_details,
            'patient_explanation': report.patient_explanation,
            'additional_notes': report.additional_notes,
            'metadata': report.metadata
        }
        
    def _format_report_as_text(self, report) -> str:
        """Format report as text."""
        sections = []
        sections.extend([
            "MEDICAL DIAGNOSTIC REPORT",
            "=" * 30,
            f"Report ID: {report.metadata['report_id']}",
            f"Date: {report.metadata['timestamp']}\n",
            "CLINICAL FINDINGS",
            "-" * 20
        ])
        
        findings = report.clinical_findings.get('observations', {})
        sections.append("Primary Findings:")
        for finding in findings.get('primary_findings', []):
            sections.append(f"- {finding}")
            
        sections.append("\nAbnormalities:")
        for abnormality in findings.get('abnormalities', []):
            sections.append(f"- {abnormality}")
        
        diagnosis = report.diagnostic_interpretation.get('primary_diagnosis', {})
        sections.extend([
            "\nDIAGNOSTIC INTERPRETATION",
            "-" * 20,
            f"Primary Diagnosis: {diagnosis.get('condition', 'Not specified')}",
            f"Confidence: {diagnosis.get('confidence', 'Not specified')}"
        ])
        
        sections.append("\nSupporting Evidence:")
        for evidence in diagnosis.get('supporting_evidence', []):
            sections.append(f"- {evidence}")
        
        sections.extend([
            "\nPATIENT EXPLANATION",
            "-" * 20
        ])
        
        explanation = report.patient_explanation
        for key, value in explanation.get('simplified_findings', {}).items():
            sections.append(f"{key}: {value}")
        
        return "\n".join(sections)

    def analyze_image(
        self,
        image: Union[str, Path, Image.Image],
        prompt: Optional[str] = None,
        generation_params: Optional[Dict] = None
    ) -> Dict:
        """Analyze a medical image and generate a report."""
        process_start = time.time()
        
        try:
            # Load and validate image
            if isinstance(image, (str, Path)):
                image = Image.open(image)
            elif not isinstance(image, Image.Image):
                raise ValueError("Image must be a file path or PIL Image")
            
            if image.mode != "RGB":
                image = image.convert("RGB")
            self.logger.info(f"Image loaded: {image.size} - Mode: {image.mode}")
            
            # Default prompt if none provided
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
            
            # Process image and generate analysis
            inputs = self.prepare_inputs_with_validation(image, prompt)
            
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
            
            gen_start = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **gen_params
                )
            
            if outputs is None or len(outputs) == 0:
                raise RuntimeError("Model generated empty output")
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            analysis = generated_text.replace(prompt, "").strip() if prompt in generated_text else generated_text.strip()
            
            # Clean up output
            sentences = analysis.split('. ')
            cleaned_sentences = []
            seen = set()
            
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and sentence not in seen:
                    cleaned_sentences.append(sentence)
                    seen.add(sentence)
            
            analysis = '. '.join(cleaned_sentences)
            if not analysis.endswith('.'):
                analysis += '.'
            
            gen_time = time.time() - gen_start
            
            # Generate structured report
            try:
                image_analysis_data = {
                    'image_type': 'x-ray',
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
                    'total_time': time.time() - process_start,
                    'prompt': prompt,
                    'success': True
                }
                
            except Exception as report_error:
                self.logger.error(f"Failed to generate structured report: {str(report_error)}")
                return {
                    'analysis': analysis,
                    'inference_time': gen_time,
                    'total_time': time.time() - process_start,
                    'prompt': prompt,
                    'success': True,
                    'report_error': str(report_error)
                }
            
        except Exception as e:
            total_time = time.time() - process_start
            self.logger.error(f"Analysis failed: {str(e)}")
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

    def analyze_image(
        self,
        image: Union[str, Path, Image.Image],
        prompt: Optional[str] = None,
        generation_params: Optional[Dict] = None
    ) -> Dict:
        """Analyze a medical image and generate a report."""
        process_start = time.time()
        
        try:
            # Load and validate image
            if isinstance(image, (str, Path)):
                image = Image.open(image)
            elif not isinstance(image, Image.Image):
                raise ValueError("Image must be a file path or PIL Image")
            
            if image.mode != "RGB":
                image = image.convert("RGB")
            self.logger.info(f"Image loaded: {image.size} - Mode: {image.mode}")
            
            # Default prompt if none provided
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
            
            # Process image and generate analysis
            inputs = self.prepare_inputs_with_validation(image, prompt)
            
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
            
            gen_start = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **gen_params
                )
            
            if outputs is None or len(outputs) == 0:
                raise RuntimeError("Model generated empty output")
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            analysis = generated_text.replace(prompt, "").strip() if prompt in generated_text else generated_text.strip()
            
            # Clean up output
            sentences = analysis.split('. ')
            cleaned_sentences = []
            seen = set()
            
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and sentence not in seen:
                    cleaned_sentences.append(sentence)
                    seen.add(sentence)
            
            analysis = '. '.join(cleaned_sentences)
            if not analysis.endswith('.'):
                analysis += '.'
            
            gen_time = time.time() - gen_start
            
            # Generate structured report
            try:
                image_analysis_data = {
                    'image_type': 'x-ray',
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
                    'total_time': time.time() - process_start,
                    'prompt': prompt,
                    'success': True
                }
                
            except Exception as report_error:
                self.logger.error(f"Failed to generate structured report: {str(report_error)}")
                return {
                    'analysis': analysis,
                    'inference_time': gen_time,
                    'total_time': time.time() - process_start,
                    'prompt': prompt,
                    'success': True,
                    'report_error': str(report_error)
                }
            
        except Exception as e:
            total_time = time.time() - process_start
            self.logger.error(f"Analysis failed: {str(e)}")
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
        """Analyze a medical image (compatibility method).
        
        This is a compatibility wrapper for analyze_image that maintains API compatibility.
        """
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