# bootstrap.py

import os
import sys
import platform
from config import logger
# import logging

# ==============================================================================
# 1. GPU Environment Setup (MUST BE DONE BEFORE IMPORTING TENSORFLOW)
# ==============================================================================

# Hardcoded fallback path
HARD_CODED_PATH_CUDA = r"E:\anaconda3\envs\shearlet_env\Library\bin"


def setup_gpu_environment():
    """
    Initialize CUDA-related paths before TensorFlow loads.
    Workflow:
        1) Windows → Try auto-locate using Python's own path (sys.prefix)
        2) If missing → fallback to HARD_CODED_PATH_CUDA
        3) Linux → Usually no action required
    """
    sys_name = platform.system()

    # === Case A: Linux (generally no manual setup required) ===
    if sys_name == "Linux":
        return

    # === Case B: Windows (must manually attach DLL directory) ===
    elif sys_name == "Windows":
        target_path = None

        # --- 1. Auto-locate Conda Library/bin via sys.prefix (Most Reliable) ---
        # sys.prefix is the root of the current environment, e.g., E:\anaconda3\envs\my_env
        candidate = os.path.join(sys.prefix, "Library", "bin")

        # [DEBUG] Print the path we are probing regardless of existence
        logger.info(f"[Bootstrap] Probing auto-detect path: {candidate}")

        if os.path.exists(candidate):
            target_path = candidate
            logger.info(f"[Bootstrap] Auto-detected CUDA path found: {target_path}")
        else:
            logger.warning(f"[Bootstrap] Auto-detected path does not exist: {candidate}")

        # --- 2. Fallback to hardcoded path (if auto-detect failed) ---
        if not target_path:
            logger.info(f"[Bootstrap] Probing fallback path: {HARD_CODED_PATH_CUDA}")

            if os.path.exists(HARD_CODED_PATH_CUDA):
                target_path = HARD_CODED_PATH_CUDA
                logger.warning(f"[Bootstrap] Auto-detect failed, using fallback: {target_path}")
            else:
                logger.error(f"[Bootstrap] Fallback CUDA path NOT found: {HARD_CODED_PATH_CUDA}")
                # Ideally, we should stop here if we know GPU is required
                raise FileNotFoundError("Critical: No CUDA path found.")

        # --- 3. Apply final resolved path ---
        if target_path:
            # Add to PATH (ensures DLL lookup by Windows loader)
            os.environ["PATH"] = target_path + os.pathsep + os.environ["PATH"]

            # Python 3.8+ explicit DLL loading
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(target_path)
                    logger.info(f"[Bootstrap] DLL directory registered successfully: {target_path}")
                except Exception as e:
                    logger.error(f"[Bootstrap] DLL registration failed: {e}")
                    raise e
        else:
            logger.error("[Bootstrap] No valid CUDA path resolved. GPU may not initialize.")
            raise SystemError


# CRITICAL: Execute setup immediately, BEFORE importing TensorFlow!
setup_gpu_environment()

# ==============================================================================
# 2. Import TensorFlow and other libraries (After Path Setup)
# ==============================================================================
import numpy as np
# Now it is safe to import TensorFlow because PATH is already updated
import tensorflow as tf
from tensorflow.keras import mixed_precision


def apply_numpy_patches():
    """
    Applies compatibility patches for legacy NumPy attributes
    removed in newer versions.
    """
    if not hasattr(np, 'NaN'): np.NaN = np.nan
    if not hasattr(np, 'float'): np.float = float
    if not hasattr(np, 'int'): np.int = int
    if not hasattr(np, 'bool'): np.bool = bool
    if not hasattr(np, 'complex'): np.complex = complex

    try:
        from numpy.testing import Tester
    except ImportError:
        # Create a dummy Tester class if missing
        class DummyTester:
            def __init__(self, *args, **kwargs): pass

            def test(self, *args, **kwargs): pass

            def bench(self, *args, **kwargs): pass

        import numpy.testing
        if not hasattr(numpy.testing, 'Tester'):
            numpy.testing.Tester = DummyTester


def initialize_tensorflow():
    """
    Initializes TensorFlow: checks GPUs and enables Mixed Precision.
    """
    logger.info("-" * 50)

    # Check visible devices
    gpus = tf.config.list_physical_devices('GPU')

    if not gpus:
        logger.warning("⚠️ No GPU detected via list_physical_devices.")
        # On Windows, sometimes physical devices list is empty but logical works,
        # but usually this means CUDA load failed.
    else:
        logger.info(f"✅ Physical GPUs detected: {len(gpus)}")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                logger.error(f"Failed to set memory growth: {e}")

    # Enable Mixed Precision (Float16) for performance
    try:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        logger.info("⚡ Mixed Precision (Float16) enabled.")
    except Exception as e:
        logger.warning(f"Could not enable mixed precision: {e}")

    logger.info("-" * 50)


# Apply patches immediately when this module is imported
apply_numpy_patches()