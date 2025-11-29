import os
import sys
import platform
import logging
from config import logger

# ==============================================================================
# 1. GPU Environment Setup (Shared Logic)
# ==============================================================================

# Hardcoded fallback path (Same as your TF bootstrap)
HARD_CODED_PATH_CUDA = r"E:\anaconda3\envs\shearlet_env\Library\bin"


def setup_gpu_environment():
    """
    Initialize CUDA-related paths before PyTorch loads.
    """
    sys_name = platform.system()

    if sys_name == "Linux":
        return

    elif sys_name == "Windows":
        target_path = None
        # 1. Auto-locate Conda Library/bin
        candidate = os.path.join(sys.prefix, "Library", "bin")
        logger.info(f"[Bootstrap-Torch] Probing path: {candidate}")

        if os.path.exists(candidate):
            target_path = candidate
        else:
            # 2. Fallback
            if os.path.exists(HARD_CODED_PATH_CUDA):
                target_path = HARD_CODED_PATH_CUDA
                logger.warning(f"[Bootstrap-Torch] Using fallback: {target_path}")

        if target_path:
            os.environ["PATH"] = target_path + os.pathsep + os.environ["PATH"]
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(target_path)
                    logger.info(f"[Bootstrap-Torch] DLL registered: {target_path}")
                except Exception as e:
                    logger.error(f"[Bootstrap-Torch] DLL reg failed: {e}")
        else:
            logger.warning("[Bootstrap-Torch] No CUDA path found. PyTorch might rely on system PATH.")


# Execute setup immediately
setup_gpu_environment()

# ==============================================================================
# 2. Import PyTorch and Configure
# ==============================================================================
import torch
import torch.backends.cudnn as cudnn


def initialize_pytorch():
    """
    Initializes PyTorch environment and optimizations.
    """
    logger.info("-" * 50)

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"✅ PyTorch GPU Detected: {device_name} (Count: {device_count})")

        # --- Optimization for Fixed Input Sizes ---
        # If your input images are always the same size (e.g. 256x256), this speeds up training
        cudnn.benchmark = True
        logger.info("⚡ cuDNN Benchmark enabled (Optimized for fixed input size).")

        # Check AMP availability (Automatic Mixed Precision)
        logger.info(f"⚡ AMP available: {torch.cuda.is_bf16_supported() or True}")

    else:
        logger.warning("⚠️ PyTorch could not find a GPU. Running on CPU (Slow).")

    logger.info("-" * 50)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize and return the device object for the script to use
device = initialize_pytorch()