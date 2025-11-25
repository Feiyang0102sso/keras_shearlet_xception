# train_phase2.py

import bootstrap  # 必须放在第一行！
import config
from config import logger, ROOT_DIR

import os
import argparse
import cv2
import numpy as np
from glob import glob
import tensorflow as tf

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

bootstrap.initialize_tensorflow()

# =================================================================
# config area
# =================================================================
DEFAULT_DATA_PATH = ROOT_DIR / 'data_processed'
DEFAULT_RESULT_PATH = ROOT_DIR / 'result_p2'
DEFAULT_PHASE1_MODEL = ROOT_DIR / 'models' / 'model_pre_best.h5'
DEFAULT_LOG_PATH = DEFAULT_RESULT_PATH / 'train_p2.log'

parser = argparse.ArgumentParser(description="Phase 2 Fine-Tuning (Partial Unfreeze)")
parser.add_argument('--dataset_root', default=DEFAULT_DATA_PATH)
parser.add_argument('--result_root', default=DEFAULT_RESULT_PATH)
parser.add_argument('--weights_path', default=DEFAULT_PHASE1_MODEL, help="Phase 1 训练好的 .h5 文件路径")

parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--batch_size', type=int, default=4, help="Batch Size (建议 8)")
parser.add_argument('--lr', type=float, default=1e-5, help="微调学习率 (必须很小，建议 1e-5)")

# 🔥 新增参数：解冻比例
parser.add_argument('--unfreeze_ratio', type=float, default=0.2,
                    help="解冻最后多少比例的层 (0.2 表示最后 20%)")


# =================================================================
# 数据管道
# =================================================================
def process_single_image_direct(path_tensor, label_tensor):
    try:
        path = path_tensor.numpy().decode('utf-8')
        label_idx = int(label_tensor.numpy())
        img = cv2.imread(path)
        if img is None: raise ValueError(f"Image not found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (299, 299))
        img_array = image.img_to_array(img)
        img_preprocessed = preprocess_input(img_array)
        label_one_hot = to_categorical(label_idx, num_classes=2)
        return img_preprocessed, label_one_hot
    except Exception as e:
        logger.error(f"Error reading {path}: {e}")
        return np.zeros((299, 299, 3), dtype=np.float32), np.zeros((2,), dtype=np.float32)


def tf_map_wrapper(path, label):
    img, lbl = tf.py_function(process_single_image_direct, [path, label], [tf.float32, tf.float32])
    img.set_shape((299, 299, 3));
    lbl.set_shape((2,))
    return img, lbl


def create_dataset(input_paths, labels, batch_size, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices((input_paths, labels))
    if is_training: dataset = dataset.shuffle(min(len(input_paths), 5000))
    dataset = dataset.map(tf_map_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


# =================================================================
# 主函数
# =================================================================
def main(args):

    config.add_file_handler(DEFAULT_LOG_PATH)

    config.ensure_dir(config.Path(args.result_root))
    if not os.path.exists(args.weights_path):
        logger.error(f"❌ 找不到权重文件: {args.weights_path}");
        return

    # --- 1. 扫描数据 ---
    logger.info("📂 正在扫描数据...")
    classes = ['FAKE', 'REAL']

    def scan(sub):
        p, l = [], []
        for i, c in enumerate(classes):
            d = os.path.join(args.dataset_root, sub, c)
            if not os.path.exists(d): d = os.path.join(args.dataset_root, sub, c.lower())
            found = glob(os.path.join(d, '*'))
            p.extend(found);
            l.extend([i] * len(found))
        return p, l

    tp, tl = scan('train');
    vp, vl = scan('test')
    if not tp: logger.error("❌ 无数据"); return

    # Shuffle
    tp = np.array(tp);
    tl = np.array(tl)
    perm = np.random.permutation(len(tp))
    tp = tp[perm];
    tl = tl[perm]
    vp = np.array(vp);
    vl = np.array(vl)

    # --- 2. 构建模型 ---
    strategy = tf.distribute.MirroredStrategy()
    logger.info(f"🚀 GPU数量: {strategy.num_replicas_in_sync}")

    with strategy.scope():
        base_model = Xception(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(2, activation='softmax', dtype='float32')(x)
        model = Model(inputs=base_model.inputs, outputs=predictions)

        logger.info(f"💾 加载 Phase 1 权重: {args.weights_path}")
        model.load_weights(args.weights_path)

        # ---------------------------------------------------------
        # 🔥 核心修改：智能解冻最后 N% 层
        # ---------------------------------------------------------
        total_layers = len(base_model.layers)
        num_unfreeze = int(total_layers * args.unfreeze_ratio)
        freeze_point = total_layers - num_unfreeze

        logger.info("-" * 40)
        logger.info(f"🧱 Xception 总层数: {total_layers}")
        logger.info(f"❄️ 冻结前 {freeze_point} 层 ({(1 - args.unfreeze_ratio) * 100:.0f}%)")
        logger.info(f"🔥 解冻后 {num_unfreeze} 层 ({args.unfreeze_ratio * 100:.0f}%)")
        logger.info("-" * 40)

        # 1. 先全部冻结
        base_model.trainable = True  # 允许模型微调
        for layer in base_model.layers:
            layer.trainable = False

        # 2. 解冻最后 N 层
        for layer in base_model.layers[freeze_point:]:
            # 💡 专家建议：保持 BatchNormalization 冻结
            # 在小 Batch Size 下微调 BN 层会破坏学到的统计数据，导致准确率下降
            if isinstance(layer, BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

        logger.info("✅ 层冻结配置完成 (BN层保持锁定)")

        model.compile(loss=categorical_crossentropy,
                      optimizer=Adam(learning_rate=args.lr),
                      metrics=['accuracy'])

    # --- 3. 训练 ---
    logger.info(f"🛡️ Batch Size: {args.batch_size}")
    train_ds = create_dataset(tp, tl, args.batch_size, True)
    val_ds = create_dataset(vp, vl, args.batch_size, False)

    save_path = os.path.join(args.result_root, 'model_final_best.h5')
    logger.info("\n🚀 [Phase 2] 开始部分微调...")

    hist = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        verbose=1,
        callbacks=[ModelCheckpoint(save_path, save_best_only=True)]
    )

    # 画图
    plt.figure()
    plt.plot(hist.history['accuracy'], label='Accuracy')
    plt.plot(hist.history['loss'], label='Loss')
    plt.legend()
    plt.savefig(os.path.join(args.result_root, 'history_phase2.png'))
    logger.info("✅ 完成！")


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)