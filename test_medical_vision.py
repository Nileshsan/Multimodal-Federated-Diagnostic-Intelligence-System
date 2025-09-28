"""
Test script for the Medical Vision Inference System with structured reporting and PDF generation.
Includes timeout handling and memory optimization.
"""

import os
import gc
import sys
import time
import torch
import signal
import threading
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, Dict, Any
from Models.vision_model.medical_inference import MedicalVisionInference
from Models.vision_model.config import SystemConfig
from Models.processors.report_generator import StructuredReportGenerator
from PIL import Image

class TimeoutException(Exception):
    """Raised when an operation times out."""
    pass

def run_with_timeout(func, args=None, kwargs=None, timeout=300):
    """Run a function with timeout using a separate thread.
    
    Args:
        func: Function to run
        args: Tuple of positional arguments
        kwargs: Dict of keyword arguments
        timeout: Timeout in seconds
    """
    args = args or ()
    kwargs = kwargs or {}
    result = []
    error = []
    
    def worker():
        try:
            result.append(func(*args, **kwargs))
        except Exception as e:
            error.append(e)
    
    thread = threading.Thread(target=worker)
    thread.daemon = True  # Daemon threads are abruptly stopped when the program exits
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        raise TimeoutException(f"Operation timed out after {timeout} seconds")
    
    if error:
        raise error[0]
        
    return result[0] if result else None

def clear_gpu_memory():
    """Clear GPU memory cache and optimize memory usage."""
    if torch.cuda.is_available():
        # Synchronize CUDA operations
        torch.cuda.synchronize()
        
        # Clear memory cache
        torch.cuda.empty_cache()
        
        # Run garbage collector multiple times
        for _ in range(3):
            gc.collect()
        
        # Force CUDA garbage collection
        with torch.cuda.device('cuda'):
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'memory_stats'):
                torch.cuda.memory_stats(device=None)
                torch.cuda.reset_peak_memory_stats()
        
        # Additional system memory cleanup
        if os.name == 'nt':  # Windows
            import ctypes
            ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1)
            
def optimize_cuda_settings():
    """Optimize CUDA settings for better performance."""
    if torch.cuda.is_available():
        # Set environment variables for better memory management
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
        
        # Enable CUDA memory caching
        torch.backends.cudnn.benchmark = True
        
        # Use deterministic algorithms for better memory usage
        torch.backends.cudnn.deterministic = True
        
        # Set conservative memory limits
        torch.cuda.set_per_process_memory_fraction(0.7)  # Use up to 70% of available GPU memory
        
        # Enable memory efficient optimizations
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def setup_output_directory(base_path="reports/medical_reports"):
    """Setup the output directory structure for medical reports."""
    current_date = datetime.now()
    year_month = current_date.strftime("%Y/%m")
    output_dir = Path(base_path) / year_month
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def generate_output_filename(diagnosis="unspecified"):
    """Generate a unique filename for the report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_diagnosis = "".join(c for c in diagnosis if c.isalnum() or c in (' ', '-', '_')).strip()
    return f"medical_report_{timestamp}_{sanitized_diagnosis}.pdf"

def print_progress(message: str, done: bool = False):
    """Print progress message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    if done:
        print(f"\n[{timestamp}] ✓ {message}")
    else:
        print(f"\n[{timestamp}] {message}", end="", flush=True)

def monitor_gpu_memory():
    """Monitor and print GPU memory usage."""
    if not torch.cuda.is_available():
        return
        
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
    print(f"\nGPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

def test_medical_vision_system():
    """Test the medical vision system with structured reporting and PDF generation."""
    
    print("\nInitializing Medical Vision System...")
    
    # Setup paths
    current_dir = Path(__file__).parent
    model_path = current_dir / "Models" / "vision_model" / "llama-3.2-11b-vision-local"
    test_image_path = current_dir / "Models" / "vision_model" / "test_images" / "pneumonia.jpg"
    log_file = current_dir / "medical_vision.log"
    
    print(f"\nInitializing Medical Vision System...")
    print(f"Model path: {model_path}")
    print(f"Test image: {test_image_path}")
    print(f"Log file: {log_file}")
    
    try:
        # Initialize the system
        config = SystemConfig()  # Using default configuration
        inference_system = MedicalVisionInference(
            model_path=model_path,
            config=config,
            log_file=log_file
        )
        
        # Load test image
        print("\nLoading test image...")
        image = Image.open(test_image_path)
        
        # Custom prompt for chest X-ray
        prompt = """
        Please analyze this chest X-ray image in detail. Provide:
        1. Image quality and positioning assessment
        2. Systematic review of anatomical structures
        3. Description of any abnormal findings
        4. Potential diagnoses based on the findings
        5. Recommendations for further evaluation if needed
        Give a detailed, professional medical assessment.
        """
        
        print("\nRunning image analysis...")
        clear_gpu_memory()  # Clear GPU memory before analysis
        
        # Set 5-minute timeout for image analysis
        # Optimize CUDA settings before analysis
        optimize_cuda_settings()
        
        print_progress("Starting image analysis (timeout: 5 minutes)")
        start_time = time.time()
        
        # Monitor memory before analysis
        print_progress("Initial GPU memory state:")
        monitor_gpu_memory()
        
        try:
            # Run analysis with timeout in separate thread
            def run_analysis():
                return inference_system.analyze_image(
                    image=image,
                    prompt=prompt
                )
            
            # First attempt
            try:
                analysis_result = run_with_timeout(run_analysis, timeout=1000)
            except (TimeoutException, Exception) as e:
                print_progress("First attempt failed, clearing memory and retrying...")
                clear_gpu_memory()
                
                # Second attempt with reduced memory usage
                torch.cuda.set_per_process_memory_fraction(0.7)  # Reduce memory usage
                try:
                    analysis_result = run_with_timeout(run_analysis, timeout=1000)
                except (TimeoutException, Exception) as e:
                    print_progress("Second attempt failed, final retry...")
                    clear_gpu_memory()
                    
                    # Final attempt with minimum memory usage
                    torch.cuda.set_per_process_memory_fraction(0.6)
                    analysis_result = run_with_timeout(run_analysis, timeout=1000)
            
            if analysis_result is None:
                raise RuntimeError("Failed to complete analysis after multiple attempts")
                
                # Calculate processing time
                processing_time = time.time() - start_time
                print_progress(f"Analysis completed in {processing_time:.1f} seconds!", done=True)
                
                # Monitor memory after analysis
                print_progress("Final GPU memory state:")
                monitor_gpu_memory()
                print("\n=== Analysis Result ===")
                if analysis_result['success']:
                    print("\n✅ Analysis completed successfully!")
                    print(f"\nInference time: {analysis_result['inference_time']:.2f} seconds")
                    print(f"Total processing time: {analysis_result['total_time']:.2f} seconds")
                    
                    # Generate report with timeout
                    try:
                        with timeout_handler(60, "Report generation timed out after 1 minute"):
                            # Set up output directory
                            output_dir = setup_output_directory()
                            output_filename = generate_output_filename()
                            output_path = output_dir / output_filename
                            
                            # Initialize report generator
                            report_generator = StructuredReportGenerator()
                            report = report_generator.generate_report(
                                analysis=analysis_result['analysis'],
                                image_path=str(test_image_path),
                                output_pdf_path=str(output_path)
                            )
                            
                            print("\n=== Report Generated ===")
                            print(f"PDF Report saved to: {output_path}")
                            
                            # Display report sections
                            if report and 'sections' in report:
                                for section, content in report['sections'].items():
                                    print(f"\n{section.upper()}:")
                                    print(content)
                    except TimeoutException:
                        print("\n⚠️ Report generation timed out!")
                        # Clean up any partial files
                        if 'output_path' in locals() and output_path.exists():
                            output_path.unlink()
                else:
                    print("\n❌ Analysis failed!")
                    print(f"Error: {analysis_result.get('error', 'Unknown error')}")
                
                # Clear GPU memory after processing
                clear_gpu_memory()
                
        except TimeoutException:
            print("\n⚠️ Image analysis timed out!")
            # Attempt to clean up
            clear_gpu_memory()
            
        except Exception as e:
            print(f"\n❌ Test failed with error: {str(e)}")
            # Ensure memory is cleared even on error
            clear_gpu_memory()
            raise
            
    except Exception as e:
        print(f"\n❌ System initialization failed: {str(e)}")
        raise

def main():
    """Main entry point with progress monitoring."""
    start_time = time.time()
    
    print_progress("Starting Medical Vision System Test")
    print("\nPress Ctrl+C at any time to abort the test and clean up resources")
    
    # Set process priority to high for better performance
    try:
        import psutil
        process = psutil.Process()
        process.nice(psutil.HIGH_PRIORITY_CLASS if os.name == 'nt' else -10)
    except Exception:
        pass  # Skip if priority setting fails
    
    try:
        # Check CUDA availability
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            print_progress(f"Using GPU: {device}")
        else:
            print_progress("WARNING: No GPU detected, using CPU", done=True)
        
        # Monitor initial memory state
        print_progress("Initial system state:")
        monitor_gpu_memory()
        
        # Run the test
        test_medical_vision_system()
        
        # Calculate total runtime
        total_time = time.time() - start_time
        print_progress(f"Test completed successfully in {total_time:.1f} seconds!", done=True)
        
    except KeyboardInterrupt:
        print_progress("\nTest aborted by user!", done=True)
        clear_gpu_memory()
        sys.exit(1)
        
    except Exception as e:
        print_progress(f"Test failed: {str(e)}", done=True)
        print_progress("Error details:", done=True)
        print(traceback.format_exc())
        clear_gpu_memory()
        raise
        
    finally:
        print_progress("Cleaning up resources")
        clear_gpu_memory()
        print_progress("Final memory state:")
        monitor_gpu_memory()

if __name__ == "__main__":
    main()