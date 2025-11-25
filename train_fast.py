# train_fast.py

# =================================================================
# 1. Bootstrap: Setup Env & DLLs (MUST BE FIRST)
# =================================================================
import bootstrap

import os
import argparse
import numpy as np
from glob import glob
import tensorflow as tf

# Matplotlib backend setting
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Keras imports
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# Local imports
import config
from config import logger, ROOT_DIR

# Initialize TensorFlow settings (Mixed Precision, etc.)
bootstrap.initialize_tensorflow()

# =================================================================
# Argument Parsing
# =================================================================
# Default to the processed data directory
DEFAULT_DATA_PATH = ROOT_DIR / 'data_processed'
DEFAULT_RESULT_PATH = ROOT_DIR / 'result_fast'
DEFAULT_LOG_PATH = DEFAULT_RESULT_PATH / 'train_fast_log'

parser = argparse.ArgumentParser(description="Fast Training on Preprocessed Data")
parser.add_argument('--dataset_root', default=DEFAULT_DATA_PATH, help="Path to preprocessed data")
parser.add_argument('--result_root', default=DEFAULT_RESULT_PATH, help="Path to output directory")
parser.add_argument('--epochs_pre', type=int, default=5, help="Epochs for Phase 1")
parser.add_argument('--epochs_fine', type=int, default=15, help="Epochs for Phase 2")
parser.add_argument('--batch_size_pre', type=int, default=16
                    , help="Per-GPU Batch Size (Phase 1)")
parser.add_argument('--batch_size_fine', type=int, default=16, help="Per-GPU Batch Size (Phase 2)")
parser.add_argument('--lr_pre', type=float, default=1e-3, help="Learning Rate (Phase 1)")
parser.add_argument('--lr_fine', type=float, default=1e-4, help="Learning Rate (Phase 2)")


# =================================================================
# 🔥 FAST Data Pipeline (Pure TensorFlow)
# No more CPU-heavy Shearlet transforms here!
# =================================================================

def load_and_preprocess_image(path, label):
    """
    Pure TensorFlow image loading pipeline.
    Extremely fast compared to python-based processing.
    """
    # 1. Read binary file
    img_raw = tf.io.read_file(path)

    # 2. Decode image (Auto-detects PNG/JPG/BMP)
    # channels=3 enforces RGB
    img = tf.image.decode_image(img_raw, channels=3, expand_animations=False)

    # 3. Resize (Ensure consistency, although preprocess.py should have handled it)
    img = tf.image.resize(img, [299, 299])

    # 4. Xception Preprocessing
    # preprocess_input expects pixel values 0-255, which decode_image provides
    img = preprocess_input(img)

    # 5. One-hot encode label
    label = tf.one_hot(label, depth=2)

    return img, label


def create_fast_dataset(input_paths, labels, global_batch_size, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices((input_paths, labels))

    if is_training:
        # Shuffle buffer size
        dataset = dataset.shuffle(buffer_size=min(len(input_paths), 10000))

    # 🔥 OPTIMIZATION: Use pure TF map, no python GIL lock
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(global_batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


# =================================================================
# Main Training Loop
# =================================================================
def main(args):
    # Ensure result directory exists
    if not os.path.exists(args.result_root):
        os.makedirs(args.result_root)

    # Setup Logger
    config.add_file_handler(DEFAULT_LOG_PATH)

    # --- 1. Init Distribution Strategy ---
    strategy = tf.distribute.MirroredStrategy()
    num_replicas = strategy.num_replicas_in_sync
    logger.info(f"\n🚀 [Distributed Training] Using {num_replicas} GPU(s).")

    # --- 2. Scaling Params ---
    global_batch_size_pre = args.batch_size_pre * num_replicas
    global_batch_size_fine = args.batch_size_fine * num_replicas
    scaled_lr_pre = args.lr_pre * num_replicas
    scaled_lr_fine = args.lr_fine * num_replicas

    logger.info(f"📊 [Scaling] Batch Size: {args.batch_size_pre} -> {global_batch_size_pre} (Global)")

    # --- 3. Scan Data ---
    classes = ['FAKE', 'REAL']  # Assumes preprocess.py kept this structure

    # Handle the fact that preprocess.py might have outputted to train/FAKE or train/fake
    # We scan generically

    logger.info(f"📂 Scanning preprocessed data in: {args.dataset_root}")

    def scan_split(split_name):
        paths, lbls = [], []
        for i, cls in enumerate(classes):
            # Try original case first, then lower case
            cls_dir = os.path.join(args.dataset_root, split_name, cls)
            if not os.path.exists(cls_dir):
                cls_dir = os.path.join(args.dataset_root, split_name, cls.lower())

            if not os.path.exists(cls_dir):
                logger.warning(f"⚠️ Directory not found: {cls_dir}")
                continue

            found = glob(os.path.join(cls_dir, '*.*'))  # Match all extensions
            paths.extend(found)
            lbls.extend([i] * len(found))
        return paths, lbls

    train_paths, train_labels = scan_split('train')
    val_paths, val_labels = scan_split('test')

    if not train_paths:
        logger.error("❌ No training images found! Check your --dataset_root.")
        return

    logger.info(f"📊 Training Samples: {len(train_paths)}")
    logger.info(f"📊 Validation Samples: {len(val_paths)}")

    # Shuffle Training Data
    # (Converting to numpy for indexing)
    train_paths = np.array(train_paths)
    train_labels = np.array(train_labels)
    perm = np.random.permutation(len(train_paths))
    train_paths = train_paths[perm]
    train_labels = train_labels[perm]

    # --- 4. Build Model ---
    logger.info("\n🏗️  Building Distributed Model...")
    with strategy.scope():
        base_model = Xception(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(len(classes), activation='softmax', dtype='float32')(x)
        model = Model(inputs=base_model.inputs, outputs=predictions)

        # Phase 1 Compile
        for layer in base_model.layers: layer.trainable = False
        model.compile(loss=categorical_crossentropy,
                      optimizer=Adam(learning_rate=scaled_lr_pre),
                      metrics=['accuracy'])

    # Create Datasets
    train_ds = create_fast_dataset(train_paths, train_labels, global_batch_size_pre, True)
    # Validation doesn't need shuffle, use numpy arrays directly
    # Note: validation paths list must be converted to tensor or array if passed directly?
    # create_fast_dataset handles list or numpy array fine.
    val_ds = create_fast_dataset(val_paths, val_labels, global_batch_size_pre, False)

    # --- 5. Training Phase 1 ---
    logger.info("\n🚀 [Phase 1] Training Head...")
    hist_pre = model.fit(
        train_ds,
        epochs=args.epochs_pre,
        validation_data=val_ds,
        verbose=1,
        callbacks=[ModelCheckpoint(os.path.join(args.result_root, 'model_pre_best.h5'), save_best_only=True)]
    )

    # --- 6. Training Phase 2 ---
    logger.info("\n🚀 [Phase 2] Fine-tuning...")
    with strategy.scope():
        for layer in model.layers: layer.trainable = True
        model.compile(loss=categorical_crossentropy,
                      optimizer=Adam(learning_rate=scaled_lr_fine),
                      metrics=['accuracy'])

    # Re-batch for fine tuning
    train_ds_fine = create_fast_dataset(train_paths, train_labels, global_batch_size_fine, True)
    val_ds_fine = create_fast_dataset(val_paths, val_labels, global_batch_size_fine, False)

    hist_fine = model.fit(
        train_ds_fine,
        epochs=args.epochs_fine,
        validation_data=val_ds_fine,
        verbose=1,
        callbacks=[ModelCheckpoint(os.path.join(args.result_root, 'model_fine_best.h5'), save_best_only=True)]
    )

    # --- 7. Save History ---
    logger.info("💾 Saving training history...")
    acc = hist_pre.history['accuracy'] + hist_fine.history['accuracy']
    loss = hist_pre.history['loss'] + hist_fine.history['loss']

    plt.figure()
    plt.plot(acc, label='Accuracy')
    plt.plot(loss, label='Loss')
    plt.legend()
    plt.savefig(os.path.join(args.result_root, 'history.png'))

    logger.info("✅ Training Finished Successfully!")


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)