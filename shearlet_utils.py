# shearlet_utils.py

import cv2
import numpy as np
from config import logger

# ================================================================
# 1. FFST Library Import (PyShearlets)
# ================================================================
try:
    import FFST

    if not hasattr(FFST, 'scalesShearsAndSpectra'):
        from FFST._scalesShearsAndSpectra import scalesShearsAndSpectra as _scalesShearsAndSpectra
        from FFST._shearletTransformSpect import shearletTransformSpect as _shearletTransformSpect
    else:
        _scalesShearsAndSpectra = FFST.scalesShearsAndSpectra
        _shearletTransformSpect = FFST.shearletTransformSpect
except (ImportError, AttributeError):
    try:
        # Fallback import attempt
        from FFST._scalesShearsAndSpectra import scalesShearsAndSpectra as _scalesShearsAndSpectra
        from FFST._shearletTransformSpect import shearletTransformSpect as _shearletTransformSpect
    except Exception:
        logger.error("❌ Critical: Failed to import FFST library.")


        def _scalesShearsAndSpectra(*args):
            return None


        def _shearletTransformSpect(*args):
            return []


# ================================================================
# 2. Wrapper Class
# ================================================================
class LocalShearletSystem:
    def __init__(self, rows, cols):
        self.st_filter = _scalesShearsAndSpectra((rows, cols))

    def coeffs(self, img_float):
        raw_output = _shearletTransformSpect(img_float, self.st_filter)

        # --- Handle inconsistent return types from PyShearlets ---

        # Case A: List containing a NumPy array (Common scenario)
        if isinstance(raw_output, list):
            if len(raw_output) == 1 and isinstance(raw_output[0], np.ndarray) and raw_output[0].ndim == 3:
                data_3d = raw_output[0]
                # Split 3D array into list of 2D arrays
                return [data_3d[:, :, i] for i in range(data_3d.shape[2])]
            return raw_output

        # Case B: Direct 3D NumPy array
        if isinstance(raw_output, np.ndarray) and raw_output.ndim == 3:
            return [raw_output[:, :, i] for i in range(raw_output.shape[2])]

        # Case C: Unknown structure (will be handled by validator downstream)
        return raw_output


# ================================================================
# 3. Processing Logic
# ================================================================
# Cache systems to avoid re-computing filters for same image dimensions
shearlet_system_cache = {}


def get_shearlet_system(rows, cols):
    if (rows, cols) in shearlet_system_cache:
        return shearlet_system_cache[(rows, cols)]
    else:
        system = LocalShearletSystem(rows=rows, cols=cols)
        shearlet_system_cache[(rows, cols)] = system
        return system


def shearlet_transform_for_cnn(image_path, target_size=(299, 299)):
    """
    Reads an image, applies Shearlet Transform, and composes a 3-channel
    image containing Low Freq, High Freq Energy, and Original content.
    """
    # --- 1. Read Image ---
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)

    t_h, t_w = target_size
    img_resized = cv2.resize(img, (t_w, t_h))
    img_float = img_resized.astype(np.float32) / 255.0

    # --- 2. Apply Transform ---
    rows, cols = img_float.shape
    ST = get_shearlet_system(rows, cols)
    coeffs_list = ST.coeffs(img_float)

    # Validate results
    is_valid = False
    if isinstance(coeffs_list, list) and len(coeffs_list) > 0:
        if isinstance(coeffs_list[0], np.ndarray) and coeffs_list[0].ndim == 2:
            is_valid = True

    if not is_valid:
        # Fallback: Return stacked original image if transform fails
        stack = np.zeros((t_h, t_w, 3), dtype=np.uint8)
        stack[:, :, 0] = img_resized
        stack[:, :, 1] = img_resized
        stack[:, :, 2] = img_resized
        return stack

    # --- 3. Compose Channels ---
    # Channel 1: Low frequency
    low_pass = np.abs(coeffs_list[0])

    # Channel 2: High frequency energy accumulation
    high_pass_energy = np.zeros_like(low_pass, dtype=np.float32)
    for band in coeffs_list[1:]:
        # Safety Resize for slight dimension mismatches
        if band.shape != high_pass_energy.shape:
            band = cv2.resize(band, (high_pass_energy.shape[1], high_pass_energy.shape[0]))
        high_pass_energy += np.abs(band)

    # Normalize to 0-255
    c1 = cv2.normalize(low_pass, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    c2 = cv2.normalize(high_pass_energy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    c3 = img_resized.astype(np.uint8)

    # --- 4. Final Safety Resize ---
    if c1.shape[:2] != (t_h, t_w): c1 = cv2.resize(c1, (t_w, t_h))
    if c2.shape[:2] != (t_h, t_w): c2 = cv2.resize(c2, (t_w, t_h))

    # --- 5. Merge ---
    final_img = np.zeros((t_h, t_w, 3), dtype=np.uint8)
    final_img[:, :, 0] = c1
    final_img[:, :, 1] = c2
    final_img[:, :, 2] = c3

    return final_img