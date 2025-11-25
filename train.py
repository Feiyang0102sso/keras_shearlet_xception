# train.py

import os
import sys
import argparse
from glob import glob

# =================================================================
# 🚨 bootstrap ust be imported here otherwise the dlls cant be find
# =================================================================
import bootstrap
import config
# must before import pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

# Keras imports
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# Local imports
from config import logger, ROOT_DIR, RESULT_PATH
import shearlet_utils

# =================================================================
# Config Area
# =================================================================

config.add_file_handler(RESULT_PATH / 'train.log')

# 2. Setup System
# utils.setup_gpu_environment()  <-- 这行其实可以注释掉，因为 import utils 时已经自动执行了
# utils.apply_numpy_patches()    <-- 这行也可以注释掉，utils 导入时也自动执行了
bootstrap.initialize_tensorflow()  # <-- 这行保留，用于打印 GPU 信息和开启混合精度

# =================================================================
# Argument Parsing
# =================================================================
parser = argparse.ArgumentParser(description="Train Xception with Shearlet Transform")
parser.add_argument('--dataset_root', default=str(config.DATA_PATH), help="Path to data directory")
parser.add_argument('--result_root', default=str(config.RESULT_PATH), help="Path to result directory")
parser.add_argument('--epochs_pre', type=int, default=5, help="Epochs for Phase 1 (frozen head)")
parser.add_argument('--epochs_fine', type=int, default=15, help="Epochs for Phase 2 (fine-tuning)")
parser.add_argument('--batch_size_pre', type=int, default=32, help="Per-GPU Batch Size for Phase 1")
parser.add_argument('--batch_size_fine', type=int, default=16, help="Per-GPU Batch Size for Phase 2")
parser.add_argument('--lr_pre', type=float, default=1e-3, help="Learning Rate for Phase 1")
parser.add_argument('--lr_fine', type=float, default=1e-4, help="Learning Rate for Phase 2")


# =================================================================
# Data Pipeline
# =================================================================

def process_single_image_py(path_tensor, label_tensor):
    """
    Python wrapper for OpenCV/Shearlet processing.
    Executes on CPU.
    """
    try:
        path = path_tensor.numpy().decode('utf-8')
        label_idx = int(label_tensor.numpy())

        # Perform complex Shearlet transform
        img = shearlet_utils.shearlet_transform_for_cnn(path, target_size=(299, 299))

        img_array = image.img_to_array(img)
        img_preprocessed = preprocess_input(img_array)
        label_one_hot = to_categorical(label_idx, num_classes=2)

        return img_preprocessed, label_one_hot
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return np.zeros((299, 299, 3), dtype=np.float32), np.zeros((2,), dtype=np.float32)


def tf_map_wrapper(path, label):
    """
    TensorFlow graph wrapper for the python function.
    """
    img, lbl = tf.py_function(
        func=process_single_image_py,
        inp=[path, label],
        Tout=[tf.float32, tf.float32]
    )
    # Explicitly set shape as py_function loses shape info
    img.set_shape((299, 299, 3))
    lbl.set_shape((2,))
    return img, lbl


def create_tf_dataset_optimized(input_paths, labels, global_batch_size, is_training=True):
    num_samples = len(input_paths)
    dataset = tf.data.Dataset.from_tensor_slices((input_paths, labels))

    if is_training:
        dataset = dataset.shuffle(buffer_size=min(num_samples, 10000))

    # num_parallel_calls=AUTOTUNE is critical for parallel CPU processing
    dataset = dataset.map(tf_map_wrapper, num_parallel_calls=tf.data.AUTOTUNE)

    # Note: batch_size here is GLOBAL (Per-GPU * num_replicas)
    dataset = dataset.batch(global_batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


# =================================================================
# Main Training Loop
# =================================================================
def main(args):
    config.ensure_dir(config.Path(args.result_root))

    # --- 1. Init Distribution Strategy ---
    strategy = tf.distribute.MirroredStrategy()
    num_replicas = strategy.num_replicas_in_sync
    logger.info(f"\n🚀 [Distributed Training] Using {num_replicas} GPU(s).")

    # --- 2. Auto-scale Batch Size and Learning Rate ---
    global_batch_size_pre = args.batch_size_pre * num_replicas
    global_batch_size_fine = args.batch_size_fine * num_replicas

    scaled_lr_pre = args.lr_pre * num_replicas
    scaled_lr_fine = args.lr_fine * num_replicas

    logger.info(f"📊 [Scaling] Batch Size: {args.batch_size_pre} -> {global_batch_size_pre} (Global)")
    logger.info(f"📊 [Scaling] Learning Rate: {args.lr_pre} -> {scaled_lr_pre}")

    # --- 3. Prepare Data ---
    classes = ['REAL', 'FAKE']
    num_classes = len(classes)
    train_dir = os.path.join(args.dataset_root, 'train')
    test_dir = os.path.join(args.dataset_root, 'test')

    logger.info("📂 Scanning files...")

    def get_file_paths(base_dir):
        inputs, targets = [], []
        for i, class_name in enumerate(classes):
            paths = []
            # Check case sensitivity (REAL vs real)
            class_path = os.path.join(base_dir, class_name)
            if not os.path.exists(class_path):
                class_path = os.path.join(base_dir, class_name.lower())

            if not os.path.exists(class_path):
                logger.warning(f"Warning: Directory not found: {class_path}")
                continue

            for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp'):
                paths.extend(glob(os.path.join(class_path, ext)))

            inputs.extend(paths)
            targets.extend([i] * len(paths))
        return inputs, targets

    train_input_paths, train_labels = get_file_paths(train_dir)
    val_input_paths, val_labels = get_file_paths(test_dir)

    if not train_input_paths:
        logger.critical(f"❌ Error: No images found in {args.dataset_root}. Check directory structure.")
        return

    logger.info(f"📊 Training Set: {len(train_input_paths)} images")
    logger.info(f"📊 Validation Set: {len(val_input_paths)} images")

    # Convert to numpy and shuffle training data
    train_input_paths = np.array(train_input_paths)
    train_labels = np.array(train_labels)
    val_input_paths = np.array(val_input_paths)
    val_labels = np.array(val_labels)

    perm_train = np.random.permutation(len(train_input_paths))
    train_input_paths = train_input_paths[perm_train]
    train_labels = train_labels[perm_train]

    # --- 4. Build Model (Within Strategy Scope) ---
    logger.info("\n🏗️  Building Distributed Model...")
    with strategy.scope():
        base_model = Xception(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(num_classes, activation='softmax', dtype='float32')(x)
        model = Model(inputs=base_model.inputs, outputs=predictions)

        # [Phase 1 Setup] Freeze base layers
        for layer in base_model.layers:
            layer.trainable = False

        model.compile(loss=categorical_crossentropy,
                      optimizer=Adam(learning_rate=scaled_lr_pre),
                      metrics=['accuracy'])

    # Create Datasets
    train_ds = create_tf_dataset_optimized(train_input_paths, train_labels, global_batch_size_pre, is_training=True)
    val_ds = create_tf_dataset_optimized(val_input_paths, val_labels, global_batch_size_pre, is_training=False)

    # --- 5. Phase 1 Training ---
    logger.info("\n🚀 [Phase 1] Training Classifier Head (Distributed)...")
    hist_pre = model.fit(
        train_ds,
        epochs=args.epochs_pre,
        validation_data=val_ds,
        verbose=1,
        callbacks=[ModelCheckpoint(os.path.join(args.result_root, 'model_pre_best.h5'), save_best_only=True)]
    )

    # --- 6. Phase 2 Fine-Tuning ---
    logger.info("\n🚀 [Phase 2] Fine-tuning (Distributed)...")

    # Re-compile within scope for fine-tuning
    with strategy.scope():
        for layer in model.layers:
            layer.trainable = True

        model.compile(optimizer=Adam(learning_rate=scaled_lr_fine),
                      loss=categorical_crossentropy,
                      metrics=['accuracy'])

    # Re-create datasets with fine-tuning batch size
    train_ds_fine = create_tf_dataset_optimized(train_input_paths, train_labels, global_batch_size_fine,
                                                is_training=True)
    val_ds_fine = create_tf_dataset_optimized(val_input_paths, val_labels, global_batch_size_fine, is_training=False)

    hist_fine = model.fit(
        train_ds_fine,
        epochs=args.epochs_fine,
        validation_data=val_ds_fine,
        verbose=1,
        callbacks=[ModelCheckpoint(os.path.join(args.result_root, 'model_fine_best.h5'), save_best_only=True)]
    )

    # --- 7. Save Results ---
    logger.info("\n💾 Saving results...")
    acc = hist_pre.history['accuracy'] + hist_fine.history['accuracy']
    loss = hist_pre.history['loss'] + hist_fine.history['loss']

    plt.figure()
    plt.plot(acc, label='Accuracy')
    plt.plot(loss, label='Loss')
    plt.title('Training History')
    plt.legend()
    plt.savefig(os.path.join(args.result_root, 'history.png'))

    logger.info("✅ All tasks completed successfully!")


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)