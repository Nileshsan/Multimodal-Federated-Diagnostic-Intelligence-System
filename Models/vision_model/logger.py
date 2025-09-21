"""Logging utilities for Medical Vision Inference system."""

import logging
import sys
from pathlib import Path
from typing import Optional
import torch

class MedicalVisionLogger:
    """Custom logger for the Medical Vision Inference system."""
    
    def __init__(self, name: str, log_file: Optional[Path] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Create formatters with safe ASCII characters
        console_formatter = logging.Formatter(
            '%(asctime)s %(message)s',
            datefmt='%H:%M:%S'
        )
        file_formatter = logging.Formatter(
            '%(asctime)s [%(name)s] %(message)s'
        )
        
        # Console handler with UTF-8 encoding for Windows
        class UTF8StreamHandler(logging.StreamHandler):
            def emit(self, record):
                try:
                    msg = self.format(record)
                    stream = self.stream
                    # Ensure proper encoding for Windows console
                    if sys.platform == 'win32':
                        try:
                            stream.buffer.write(msg.encode('utf-8'))
                            stream.buffer.write(self.terminator.encode('utf-8'))
                        except AttributeError:
                            stream.write(msg + self.terminator)
                    else:
                        stream.write(msg + self.terminator)
                    self.flush()
                except Exception:
                    self.handleError(record)
        
        # Use our custom handler
        console_handler = UTF8StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def log_system_info(self):
        """Log system information including GPU status."""
        import psutil
        import GPUtil
        
        self.logger.info("=" * 50)
        self.logger.info("System Information:")
        
        # CPU Info
        self.logger.info(f"CPU Usage: {psutil.cpu_percent()}%")
        self.logger.info(f"Memory Usage: {psutil.virtual_memory().percent}%")
        
        # GPU Info
        if torch.cuda.is_available():
            gpu = GPUtil.getGPUs()[0]
            self.logger.info(f"GPU: {gpu.name}")
            self.logger.info(f"GPU Memory: {gpu.memoryUsed}/{gpu.memoryTotal} MB")
            self.logger.info(f"GPU Load: {gpu.load * 100:.1f}%")
    
    def log_memory_stats(self):
        """Log current GPU memory statistics."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            self.logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    def debug(self, msg: str):
        """Log debug level message."""
        self.logger.debug(msg)
    
    def info(self, msg: str):
        """Log info message."""
        self.logger.info(msg)
    
    def warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)
    
    def error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)
    
    def success(self, msg: str):
        """Log success message."""
        self.logger.info(f"✓ {msg}")  # Using ✓ instead of ✅ for better compatibility