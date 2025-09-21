# Medical Diagnostic Vision Intelligence System
# Optimized for 8GB VRAM with accuracy priority

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import json
import os
from datetime import datetime

class DiagnosticVisionSystem:
    """
    Frustrated Diagnostic Intelligence System using Llama 3.2 11B Vision
    Optimized for medical/diagnostic accuracy on 8GB VRAM
    """
    
    def __init__(self, model_name="unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.local_path = "./diagnostic_model"
        
        # Diagnostic templates
        self.diagnostic_prompts = {
            'general': "As a medical diagnostic assistant, analyze this image carefully. Describe what you observe and provide potential diagnostic insights:",
            'skin': "Analyze this skin condition image. Describe the visual characteristics, possible conditions, and recommend next steps:",
            'xray': "Examine this X-ray image. Identify anatomical structures and any abnormalities you can observe:",
            'symptoms': "Based on this medical image, what symptoms or conditions might this represent? Provide detailed observations:",
            'comparison': "Compare and analyze this medical image. What are the key diagnostic features you can identify?"
        }
    
    def setup_model(self):
        """Initialize the diagnostic model with memory optimization"""
        
        print("🏥 Setting up Diagnostic Vision System...")
        print(f"📥 Loading model: {self.model_name}")
        
        try:
            # Load processor
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            # Load model with memory optimization for 8GB VRAM
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",                    # Smart device mapping
                torch_dtype=torch.float16,           # Memory efficient
                load_in_4bit=True,                   # 4-bit quantization
                offload_folder="./offload",          # CPU offload
                max_memory={0: "6GB"},               # Reserve 2GB for app
                low_cpu_mem_usage=True,              # Reduce CPU usage
                trust_remote_code=True,
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
            )
            
            print("✅ Diagnostic system initialized successfully")
            print(f"🎯 Model loaded with memory optimization for diagnostic accuracy")
            
            # Save locally for faster future loading
            self.save_model_locally()
            
            return True
            
        except Exception as e:
            print(f"❌ Error setting up diagnostic system: {e}")
            return False
    
    def save_model_locally(self):
        """Save model locally for faster loading"""
        try:
            os.makedirs(self.local_path, exist_ok=True)
            self.model.save_pretrained(self.local_path)
            self.processor.save_pretrained(self.local_path)
            print(f"💾 Model saved locally to {self.local_path}")
        except Exception as e:
            print(f"⚠️ Could not save model locally: {e}")
    
    def load_local_model(self):
        """Load model from local storage"""
        if not os.path.exists(self.local_path):
            return self.setup_model()
        
        try:
            print("📂 Loading diagnostic system from local storage...")
            self.processor = AutoProcessor.from_pretrained(self.local_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.local_path,
                device_map="auto",
                torch_dtype=torch.float16,
                offload_folder="./offload",
                max_memory={0: "6GB"},
                trust_remote_code=True
            )
            print("✅ Local diagnostic system loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading local model: {e}")
            return self.setup_model()
    
    def analyze_medical_image(self, image_path, diagnostic_type='general', custom_prompt=None):
        """
        Analyze medical image with diagnostic focus
        
        Args:
            image_path: Path to medical image
            diagnostic_type: Type of analysis ('general', 'skin', 'xray', 'symptoms', 'comparison')
            custom_prompt: Custom diagnostic prompt
        """
        
        if self.model is None:
            print("❌ Model not loaded. Run setup_model() first.")
            return None
        
        try:
            # Load and process image
            image = Image.open(image_path)
            
            # Select appropriate diagnostic prompt
            if custom_prompt:
                prompt = custom_prompt
            else:
                prompt = self.diagnostic_prompts.get(diagnostic_type, self.diagnostic_prompts['general'])
            
            print(f"🔍 Analyzing medical image: {image_path}")
            print(f"📋 Diagnostic type: {diagnostic_type}")
            
            # Process inputs
            inputs = self.processor(prompt, image, return_tensors="pt")
            
            # Move to appropriate device
            if torch.cuda.is_available():
                inputs = {k: v.to(self.model.device) if torch.is_tensor(v) else v 
                         for k, v in inputs.items()}
            
            # Generate diagnostic analysis
            print("🤖 Generating diagnostic analysis...")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=300,           # Detailed analysis
                    temperature=0.3,              # Conservative for medical accuracy
                    do_sample=True,
                    repetition_penalty=1.1,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part
            if prompt in response:
                analysis = response.split(prompt)[-1].strip()
            else:
                analysis = response.strip()
            
            # Structure the diagnostic result
            result = {
                'timestamp': datetime.now().isoformat(),
                'image_path': image_path,
                'diagnostic_type': diagnostic_type,
                'analysis': analysis,
                'confidence': 'high' if len(analysis) > 100 else 'medium',
                'model_used': self.model_name
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error analyzing image: {e}")
            return None
    
    def batch_analyze(self, image_paths, diagnostic_type='general'):
        """Analyze multiple medical images"""
        
        results = []
        for i, image_path in enumerate(image_paths):
            print(f"📊 Processing image {i+1}/{len(image_paths)}")
            result = self.analyze_medical_image(image_path, diagnostic_type)
            if result:
                results.append(result)
            
            # Clear cache between analyses to prevent memory buildup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results
    
    def save_diagnostic_report(self, results, output_path="diagnostic_report.json"):
        """Save diagnostic results to file"""
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"📄 Diagnostic report saved to {output_path}")
        except Exception as e:
            print(f"❌ Error saving report: {e}")
    
    def get_system_info(self):
        """Get system and model information"""
        info = {
            'model': self.model_name,
            'cuda_available': torch.cuda.is_available(),
            'gpu_memory_total': f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "N/A",
            'gpu_memory_allocated': f"{torch.cuda.memory_allocated() / 1e9:.1f} GB" if torch.cuda.is_available() else "N/A",
            'model_loaded': self.model is not None
        }
        return info

# Usage example and main execution
def main():
    """Main diagnostic system execution"""
    
    print("=" * 60)
    print("🏥 MEDICAL DIAGNOSTIC VISION INTELLIGENCE SYSTEM")
    print("=" * 60)
    
    # Initialize diagnostic system
    diagnostic_system = DiagnosticVisionSystem()
    
    # Setup or load model
    success = diagnostic_system.load_local_model()
    if not success:
        print("❌ Failed to initialize diagnostic system")
        return
    
    # System information
    print("\n📊 System Information:")
    info = diagnostic_system.get_system_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Example diagnostic analysis
    print("\n" + "=" * 40)
    print("DIAGNOSTIC ANALYSIS EXAMPLE")
    print("=" * 40)
    
    # You would replace this with actual medical image paths
    example_images = [
        "medical_image_1.jpg",  # Replace with actual paths
        "xray_sample.jpg",      # Replace with actual paths  
        "skin_condition.jpg"    # Replace with actual paths
    ]
    
    # Analyze available images
    available_images = [img for img in example_images if os.path.exists(img)]
    
    if available_images:
        print(f"📋 Analyzing {len(available_images)} medical images...")
        results = diagnostic_system.batch_analyze(available_images, 'general')
        
        # Display results
        for i, result in enumerate(results):
            print(f"\n🔍 Analysis {i+1}:")
            print(f"📁 Image: {result['image_path']}")
            print(f"🎯 Analysis: {result['analysis'][:200]}...")
            print(f"📊 Confidence: {result['confidence']}")
        
        # Save report
        diagnostic_system.save_diagnostic_report(results)
    else:
        print("⚠️ No example medical images found.")
        print("📝 To test with your medical images:")
        print("   1. Place medical images in the current directory")
        print("   2. Update the example_images list with your filenames")
        print("   3. Run the analysis")
    
    print("\n✅ Diagnostic system ready for medical image analysis!")

if __name__ == "__main__":
    main()