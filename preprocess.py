import bootstrap  # Must be the first import to setup DLLs/Env

import os
import argparse
import cv2
import glob
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np

# Import your custom modules
import shearlet_utils
from config import logger


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def process_and_save_single_image(args):
    """
    Worker function to process a single image.
    Args:
        args: tuple (src_path, dst_path)
    """
    src_path, dst_path = args

    # 1. Resume capability: Skip if target already exists
    if os.path.exists(dst_path):
        return

    try:
        # 2. Perform Shearlet Transform (CPU intensive)
        # Returns RGB format
        processed_img = shearlet_utils.shearlet_transform_for_cnn(src_path, target_size=(299, 299))

        # 3. Convert RGB -> BGR for OpenCV
        # Keras preprocessing usually yields RGB, but OpenCV saves as BGR.
        # We convert it back to BGR to ensure correct colors when saving.
        img_bgr = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)

        # 4. Save to disk
        cv2.imwrite(dst_path, img_bgr)

    except Exception as e:
        # Catch errors to prevent crashing the whole pool
        print(f"[Error] Failed to process {src_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Offline Shearlet Preprocessing")
    parser.add_argument('--src_root', default='./data', help="Path to source data directory")
    parser.add_argument('--dst_root', default='./data_processed', help="Path to output directory")

    # Dynamic worker calculation:
    # Leave 2 cores free for OS responsiveness, but ensure at least 1 worker.
    cpu_count = os.cpu_count() or 1
    safe_defaults = max(1, cpu_count - 2)

    parser.add_argument('--workers', type=int, default=safe_defaults,
                        help=f"Number of parallel processes (Default on this machine: {safe_defaults})")

    args = parser.parse_args()

    logger.info("🚀 [Preprocessing] Starting offline transformation...")
    logger.info(f"📂 Source: {args.src_root}")
    logger.info(f"📂 Dest  : {args.dst_root}")
    logger.info(f"🔥 Workers: {args.workers} (detected {os.cpu_count()} logical cores)")

    # Define sub-directories to scan
    # Adjust these if your folder structure is different
    sub_dirs = ['train/FAKE', 'train/REAL', 'test/FAKE', 'test/REAL']
    tasks = []

    # --- 1. Scan files and build task list ---
    for sub in sub_dirs:
        # Attempt to find source directory (Case insensitive logic for Windows/Linux compatibility)
        src_dir = os.path.join(args.src_root, sub)

        # Try lowercase if not found (e.g., train/fake)
        if not os.path.exists(src_dir):
            src_dir = os.path.join(args.src_root, sub.lower())

        # Try mixed case if not found (e.g., train/FAKE)
        if not os.path.exists(src_dir):
            parts = sub.split('/')
            if len(parts) > 1:
                src_dir = os.path.join(args.src_root, parts[0], parts[1].upper())

        if not os.path.exists(src_dir):
            logger.warning(f"⚠️  Directory not found, skipping: {sub}")
            continue

        # Prepare destination directory
        dst_dir = os.path.join(args.dst_root, sub)
        ensure_dir(dst_dir)

        # Scan images
        files = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
            files.extend(glob.glob(os.path.join(src_dir, ext)))

        # Add to task list
        for f in files:
            file_name = os.path.basename(f)
            dst_f = os.path.join(dst_dir, file_name)
            tasks.append((f, dst_f))

    total_files = len(tasks)
    if total_files == 0:
        logger.error("❌ No images found! Check your source directory structure.")
        return

    logger.info(f"📊 Found {total_files} images. Beginning processing...")

    # --- 2. Execute with Multiprocessing ---
    # Using ProcessPoolExecutor to bypass Python GIL for true parallelism
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            # list() forces execution; tqdm provides the progress bar
            list(tqdm(
                executor.map(process_and_save_single_image, tasks),
                total=total_files,
                unit="img",
                desc="Processing"
            ))
    else:
        # Fallback for single-core debugging
        for task in tqdm(tasks, total=total_files, unit="img", desc="Processing"):
            process_and_save_single_image(task)

    logger.info("✅ Preprocessing completed successfully!")
    logger.info(f"💾 Processed data saved to: {args.dst_root}")


if __name__ == '__main__':
    # Essential for Windows multiprocessing support
    main()