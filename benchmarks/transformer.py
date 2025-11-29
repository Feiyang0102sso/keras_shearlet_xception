import os
import time
import argparse
import copy
import sys
from tqdm import tqdm

# fix dir problems
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


# --- 1. Bootstrap Environment (PyTorch) ---
# import bootstrap_pytorch
from bootstrap_pytorch import device

# --- 2. Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

from config import Path, add_file_handler, ensure_dir, logger, ROOT_DIR

# =================================================================
# config area
# =================================================================
DEFAULT_DATA_PATH = ROOT_DIR / 'data'
DEFAULT_RESULT_PATH = ROOT_DIR / 'result_benchmark'
DEFAULT_LOG_PATH = DEFAULT_RESULT_PATH / 'train_benchmark.log'


# ==============================================================================
# 3. Model Definition (Original LCVT from Paper)
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
        # Ensure we use standard MobileNet weights
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
        x = self.pool(x)  # B x 1280 x 16 x 16
        x = self.proj(x)  # B x 256 x 16 x 16
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  # B x 256 -> sequence of patches
        x = self.transformer(x)  # B x (H*W) x 256
        x = x.mean(dim=1)  # Global average pooling
        x = self.head(x)  # Final classification
        return x


# ==============================================================================
# 4. Data Pipeline
# ==============================================================================
def get_dataloaders(dataset_root, batch_size, img_size=256):
    """
    dataset_root: 传入的参数路径 (Path对象 或 str)
    """
    # 确保是 Path 对象
    data_dir = Path(dataset_root)

    # 适配你的目录结构: train / test
    train_dir = str(data_dir / 'train')
    val_dir = str(data_dir / 'test')

    if not os.path.exists(train_dir):
        logger.critical(f"❌ Dataset not found at: {train_dir}")
        raise FileNotFoundError(f"Check path: {train_dir}")

    # Image Transformations
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    # 打印映射关系
    logger.info(f"🏷️ Class Mapping: {train_dataset.class_to_idx}")
    logger.info(f"📊 Train Images: {len(train_dataset)}")
    logger.info(f"📊 Test Images:  {len(val_dataset)}")

    # workers = 4 if os.name != 'nt' else 0
    workers = 2

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
    #                           num_workers=workers, pin_memory=True)  # 去掉 persistent_workers
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
    #                         num_workers=workers, pin_memory=True)

    return train_loader, val_loader, len(train_dataset), len(val_dataset)


# ==============================================================================
# 5. Training Loop (已修改支持自定义路径)
# ==============================================================================
# add save_dir
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, save_dir, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # 使用 Mixed Precision (AMP) 加速训练
    scaler = torch.amp.GradScaler('cuda')

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

            pbar = tqdm(dataloaders[phase], desc=f"{phase} Phase", unit="batch", leave=False)

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    # AMP Context
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    # Backward + Optimize only if in training phase
                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            logger.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                target_path = Path(save_dir) / 'lcvt_torch_best.pth'

                torch.save(model.state_dict(), str(target_path))
                logger.info(f"💾 Saved new best model to {target_path}")

    time_elapsed = time.time() - since
    logger.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logger.info(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model


# ==============================================================================
# 6. Main Execution
# ==============================================================================
def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Train PyTorch LCVT Benchmark Model")

    # Path Arguments
    parser.add_argument('--dataset_root', default=DEFAULT_DATA_PATH, type=Path,
                        help=f"数据根目录 (默认: {DEFAULT_DATA_PATH})")
    parser.add_argument('--result_root', default=DEFAULT_RESULT_PATH, type=Path,
                        help=f"结果保存目录 (默认: {DEFAULT_RESULT_PATH})")
    parser.add_argument('--log_path', default=DEFAULT_LOG_PATH, type=Path,
                        help="日志文件路径")

    # Hyperparameter Arguments
    parser.add_argument('--epochs', type=int, default=30, help="训练轮数")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch Size")
    parser.add_argument('--lr', type=float, default=1e-4, help="学习率 (建议 1e-4)")

    args = parser.parse_args()

    # --- Setup System ---
    # 1. Ensure Result Directory Exists
    ensure_dir(args.result_root)

    # 2. Setup Logging to File
    # 如果用户传入的 log_path 是相对路径，它会自动基于 result_root 或当前目录解析
    add_file_handler(str(args.log_path))

    logger.info("🚀 Starting Benchmark Training (PyTorch)")
    logger.info(f"📂 Data Root:   {args.dataset_root}")
    logger.info(f"📂 Result Dir:  {args.result_root}")
    logger.info(f"⚙️ Config:      Epochs={args.epochs}, Batch={args.batch_size}, LR={args.lr}")

    # --- Prepare Data ---
    # 将参数传入 get_dataloaders
    train_loader, val_loader, train_len, val_len = get_dataloaders(args.dataset_root, args.batch_size)
    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': train_len, 'val': val_len}

    # --- Initialize Model ---
    model = ForgeryDetector(num_classes=2)
    model = model.to(device)

    # --- Optimizer & Loss ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # --- Start Training ---
    try:
        model = train_model(
            model,
            dataloaders,
            dataset_sizes,
            criterion,
            optimizer,
            save_dir=args.result_root,
            num_epochs=args.epochs
        )

        # --- Save Final Model ---
        final_path = args.result_root / 'lcvt_torch_final.pth'
        torch.save(model.state_dict(), str(final_path))
        logger.info(f"✅ Training Finished. Model saved to: {final_path}")

    except KeyboardInterrupt:
        logger.warning("⚠️ Training interrupted by user.")
        save_path = args.result_root / 'lcvt_torch_interrupted.pth'
        torch.save(model.state_dict(), str(save_path))
        logger.info(f"💾 Emergency save: {save_path}")
    except Exception as e:
        logger.critical(f"❌ Critical Error during training: {e}", exc_info=True)


if __name__ == '__main__':
    main()