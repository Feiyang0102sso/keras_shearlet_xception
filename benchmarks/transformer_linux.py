import os
import time
import argparse
import copy
import sys
import multiprocessing
from tqdm import tqdm

# fix dir problems
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# --- 1. Bootstrap Environment ---
from bootstrap_pytorch import device

# --- 2. Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

from config import Path, add_file_handler, ensure_dir, logger, ROOT_DIR

# 开启 cuDNN 自动寻优 (6x4090 必开)
torch.backends.cudnn.benchmark = True

# =================================================================
# config area
# =================================================================
DEFAULT_DATA_PATH = ROOT_DIR / 'data'
DEFAULT_RESULT_PATH = ROOT_DIR / 'result_benchmark'
DEFAULT_LOG_PATH = DEFAULT_RESULT_PATH / 'train_benchmark.log'


# ==============================================================================
# 3. Model Definition
# ==============================================================================
class ConvTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        return x


class ForgeryDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features
        self.pool = nn.AdaptiveAvgPool2d((16, 16))
        self.proj = nn.Conv2d(1280, 256, kernel_size=1)
        self.transformer = nn.Sequential(
            ConvTransformerBlock(256, 4),
            ConvTransformerBlock(256, 4)
        )
        self.head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x


# ==============================================================================
# 4. Data Pipeline
# ==============================================================================
def get_dataloaders(dataset_root, batch_size, img_size=256, num_workers=16):
    data_dir = Path(dataset_root)
    train_dir = str(data_dir / 'train')
    val_dir = str(data_dir / 'test')

    if not os.path.exists(train_dir):
        logger.critical(f"❌ Dataset not found at: {train_dir}")
        raise FileNotFoundError(f"Check path: {train_dir}")

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    logger.info(f"🏷️ Class Mapping: {train_dataset.class_to_idx}")
    logger.info(f"📊 Train Images: {len(train_dataset)}")
    logger.info(f"📊 Test Images:  {len(val_dataset)}")
    logger.info(f"🚀 Data Loader: Workers={num_workers}, Pin_Memory=True")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    return train_loader, val_loader, len(train_dataset), len(val_dataset)


# ==============================================================================
# 5. Training Loop (Multi-GPU Support)
# ==============================================================================
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, save_dir, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(
        model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict())
    best_acc = 0.0

    # 自动适配 Mixed Precision
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch + 1}/{num_epochs}')
        logger.info('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # 动态调整进度条宽度
            pbar = tqdm(
                dataloaders[phase],
                desc=f"{phase:<5}",
                unit="batch",
                leave=True,
                dynamic_ncols=True,
                mininterval=0.2
            )

            for inputs, labels in pbar:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                # Gather loss from all GPUs (loss is averaged by default in DP)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # 计算当前平均 Loss
                current_avg_loss = running_loss / ((pbar.n + 1) * inputs.size(0))

                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg': f"{current_avg_loss:.4f}"
                })

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            logger.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 保存最优模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # 注意：DataParallel 需要保存 module 的参数
                state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                best_model_wts = copy.deepcopy(state_dict)

                target_path = Path(save_dir) / 'lcvt_torch_best.pth'
                torch.save(state_dict, str(target_path))
                logger.info(f"💾 Saved new best model to {target_path}")

    time_elapsed = time.time() - since
    logger.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logger.info(f'Best val Acc: {best_acc:4f}')

    return model


# ==============================================================================
# 6. Main Execution
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train PyTorch LCVT (6x RTX 4090 Monster Mode)")

    parser.add_argument('--dataset_root', default=DEFAULT_DATA_PATH, type=Path)
    parser.add_argument('--result_root', default=DEFAULT_RESULT_PATH, type=Path)
    parser.add_argument('--log_path', default=DEFAULT_LOG_PATH, type=Path)

    # 4090 Monster 参数设置
    parser.add_argument('--epochs', type=int, default=50)
    # 6张卡，建议总 Batch Size 设为 512 或 768
    parser.add_argument('--batch_size', type=int, default=512, help="建议: 256 或 512 (6卡)")
    parser.add_argument('--lr', type=float, default=1e-4)

    # 你的 CPU 有 88 个线程，可以开 32 个 Worker 喂数据
    parser.add_argument('--workers', type=int, default=32, help="数据读取进程数 (建议 16-32)")

    args = parser.parse_args()

    ensure_dir(args.result_root)
    add_file_handler(str(args.log_path))

    # 检测显卡数量
    gpu_count = torch.cuda.device_count()
    logger.info("=" * 50)
    logger.info(f"🚀 Detected {gpu_count}x GPUs! Activating Monster Mode.")
    logger.info("=" * 50)

    train_loader, val_loader, train_len, val_len = get_dataloaders(
        args.dataset_root,
        args.batch_size,
        num_workers=args.workers
    )
    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': train_len, 'val': val_len}

    model = ForgeryDetector(num_classes=2)
    model = model.to(device)

    # 🔥 核心：开启多卡并行 DataParallel
    if gpu_count > 1:
        model = nn.DataParallel(model)
        logger.info(f"⚡ Model distributed across {gpu_count} GPUs.")

    logger.info(f"⚙️ Config: Epochs={args.epochs}, Total Batch={args.batch_size}, Workers={args.workers}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    try:
        train_model(
            model,
            dataloaders,
            dataset_sizes,
            criterion,
            optimizer,
            save_dir=args.result_root,
            num_epochs=args.epochs
        )

        # 保存最终模型
        final_path = args.result_root / 'lcvt_torch_final.pth'
        state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(state_dict, str(final_path))
        logger.info(f"✅ Training Finished. Model saved to: {final_path}")

    except KeyboardInterrupt:
        logger.warning("⚠️ Training interrupted.")
    except Exception as e:
        logger.critical(f"❌ Error: {e}", exc_info=True)


if __name__ == '__main__':
    main()