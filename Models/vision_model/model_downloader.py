"""
Llama 3.2 11B Vision Model Downloader
Downloads and saves the model locally for medical diagnostic vision tasks
"""

import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from huggingface_hub import snapshot_download, HfFolder
import psutil
import GPUtil

class ModelDownloader:
    """Download and setup Llama 3.2 11B Vision model for medical diagnostics"""
    
    def __init__(self):
        self.model_name = "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit"
        # Store model in the vision_model directory
        self.local_path = os.path.join(os.path.dirname(__file__), "llama-3.2-11b-vision-local")
        self.download_stats = {}
        
    def check_system_requirements(self):
        """Check system specs before download"""
        print("🔍 Checking System Requirements...")
        print("=" * 50)
        
        # CPU Info
        cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        print(f"💻 CPU: {cpu_count} cores")
        if cpu_freq:
            print(f"💻 CPU Frequency: {cpu_freq.current:.2f} MHz")
        
        # RAM Info
        ram = psutil.virtual_memory()
        print(f"🧠 RAM: {ram.total / (1024**3):.1f} GB total, {ram.available / (1024**3):.1f} GB available")
        
        # GPU Info
        if torch.cuda.is_available():
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                print(f"🎮 GPU: {gpu.name}")
                print(f"🎮 VRAM: {gpu.memoryTotal} MB total, {gpu.memoryFree} MB free")
                print(f"🎮 GPU Usage: {gpu.load * 100:.1f}%")
        else:
            print("❌ CUDA not available - will download but inference will be slow")
        
        # Disk Space
        disk = psutil.disk_usage('.')
        print(f"💾 Disk Space: {disk.free / (1024**3):.1f} GB free")
        
        print("=" * 50)
        
        # Requirements check
        requirements_met = True
        if ram.available < 16 * (1024**3):  # Less than 16GB RAM
            print("⚠️  Warning: Less than 16GB RAM available. Download may be slow.")
        
        if disk.free < 20 * (1024**3):  # Less than 20GB free space
            print("❌ Error: Need at least 20GB free disk space")
            requirements_met = False
        
        if not torch.cuda.is_available():
            print("⚠️  Warning: CUDA not available. Inference will use CPU (very slow)")
        
        return requirements_met
    
    def check_existing_model(self):
        """Check if model already exists locally"""
        if os.path.exists(self.local_path):
            print(f"📁 Found existing model at: {self.local_path}")
            
            # Check if it's complete
            required_files = ['config.json', 'pytorch_model.bin.index.json', 'tokenizer.json']
            missing_files = []
            
            for file in required_files:
                if not os.path.exists(os.path.join(self.local_path, file)):
                    missing_files.append(file)
            
            if missing_files:
                print(f"❌ Incomplete download. Missing: {missing_files}")
                return False
            else:
                print("✅ Complete model found locally")
                return True
        return False
    
    def download_with_progress(self):
        """Download model with progress tracking"""
        print(f"\n🚀 Starting download of {self.model_name}")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Create local directory
            os.makedirs(self.local_path, exist_ok=True)
            
            # Download using snapshot_download for better progress tracking
            print("📥 Downloading model files...")
            
            downloaded_path = snapshot_download(
                repo_id=self.model_name,
                local_dir=self.local_path,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            
            download_time = time.time() - start_time
            
            print(f"✅ Download completed!")
            print(f"⏱️  Download time: {download_time / 60:.1f} minutes")
            print(f"📁 Model saved to: {downloaded_path}")
            
            self.download_stats = {
                'download_time_minutes': download_time / 60,
                'local_path': downloaded_path,
                'success': True
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            self.download_stats = {
                'error': str(e),
                'success': False
            }
            return False
    
    def verify_download(self):
        """Verify the downloaded model"""
        print("\n🔍 Verifying downloaded model...")
        
        try:
            # Try to load tokenizer
            print("📝 Testing tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(self.local_path)
            print("✅ Tokenizer loaded successfully")
            
            # Try to load processor
            print("🖼️  Testing processor...")
            processor = AutoProcessor.from_pretrained(self.local_path)
            print("✅ Processor loaded successfully")
            
            print("✅ Model verification completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Verification failed: {e}")
            return False
    
    def get_model_info(self):
        """Get detailed model information"""
        if not os.path.exists(self.local_path):
            print("❌ Model not found locally")
            return None
        
        try:
            # Load config
            import json
            config_path = os.path.join(self.local_path, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                info = {
                    'model_type': config.get('model_type', 'Unknown'),
                    'vocab_size': config.get('vocab_size', 'Unknown'),
                    'hidden_size': config.get('hidden_size', 'Unknown'),
                    'num_hidden_layers': config.get('num_hidden_layers', 'Unknown'),
                    'num_attention_heads': config.get('num_attention_heads', 'Unknown'),
                    'max_position_embeddings': config.get('max_position_embeddings', 'Unknown'),
                }
                
                print("\n📋 Model Information:")
                print("=" * 30)
                for key, value in info.items():
                    print(f"{key:25}: {value}")
                
                return info
        except Exception as e:
            print(f"❌ Could not read model config: {e}")
        
        return None

def main():
    """Main download execution"""
    print("=" * 70)
    print("🦙 LLAMA 3.2 11B VISION MODEL DOWNLOADER")
    print("=" * 70)
    
    downloader = ModelDownloader()
    
    # Step 1: Check system requirements
    if not downloader.check_system_requirements():
        print("❌ System requirements not met. Exiting.")
        return
    
    # Step 2: Check if model already exists
    if downloader.check_existing_model():
        choice = input("\n📁 Model already exists. Re-download? (y/N): ").lower()
        if choice != 'y':
            print("✅ Using existing model")
            downloader.get_model_info()
            return
    
    # Step 3: Start download
    print("\n🚀 Starting download...")
    success = downloader.download_with_progress()
    
    if not success:
        print("❌ Download failed. Check your internet connection and try again.")
        return
    
    # Step 4: Verify download
    if downloader.verify_download():
        print("\n🎉 SUCCESS! Model downloaded and verified")
        downloader.get_model_info()
        
        print("\n" + "=" * 50)
        print("✅ READY FOR INFERENCE!")
        print("📁 Model location:", downloader.local_path)
        print("=" * 50)
    else:
        print("❌ Download verification failed")

if __name__ == "__main__":
    main()