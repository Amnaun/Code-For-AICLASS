import contextlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CONFIG: Dict[str, object] = {
    "data_root": Path("dataset/scenes").expanduser(),
    "work_dir": Path("work"),
    "train_txt": Path("work/train.txt"),
    "eval_txt": Path("work/eval.txt"),
    "readme_json": Path("work/readme.json"),
    "input_size": 224,
    "val_ratio": 0.125,
    "batch_size": 32,
    "eval_batch_size": 64,
    "num_epochs": 12,
    "learning_rate": 5e-4,
    "weight_decay": 1e-2,
    "label_smoothing": 0.05,
    "num_workers": 4,
    "max_grad_norm": 5.0,
    "warmup_pct": 0.15,
    "seed": 2024,
}
CONFIG["checkpoint_dir"] = CONFIG["work_dir"] / "checkpoints"
CONFIG["best_model_path"] = CONFIG["checkpoint_dir"] / "scene_net_best.pt"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dataset(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"数据集路径不存在: {path}")
    if not any(path.iterdir()):
        raise RuntimeError(f"数据集为空: {path}")


def list_image_samples(root: Path) -> Tuple[List[Tuple[Path, int]], List[str]]:
    samples: List[Tuple[Path, int]] = []
    classes = [
        d for d in sorted(root.iterdir()) if d.is_dir() and d.name != ".DS_Store"
    ]
    if not classes:
        raise RuntimeError(f"未在 {root} 下找到类别文件夹")
    for label, class_dir in enumerate(classes):
        for image_path in sorted(class_dir.glob("*")):
            if image_path.suffix.lower() in IMG_EXTENSIONS:
                samples.append((image_path, label))
    class_names = [c.name for c in classes]
    if not samples:
        raise RuntimeError("未找到任何图像文件")
    return samples, class_names


def stratified_split(
    samples: Sequence[Tuple[Path, int]], val_ratio: float, seed: int
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    grouped: Dict[int, List[Tuple[Path, int]]] = defaultdict(list)
    for sample in samples:
        grouped[sample[1]].append(sample)
    train_samples: List[Tuple[Path, int]] = []
    val_samples: List[Tuple[Path, int]] = []
    rng = random.Random(seed)
    for _, rows in grouped.items():
        rng.shuffle(rows)
        val_count = max(1, int(len(rows) * val_ratio))
        val_rows = rows[:val_count]
        train_rows = rows[val_count:]
        if not train_rows:
            train_rows = [val_rows.pop()]
        val_samples.extend(val_rows)
        train_samples.extend(train_rows)
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def export_splits(samples: Sequence[Tuple[Path, int]], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for path, label in samples:
            f.write(f"{path}\t{label}\n")


def save_readme(
    samples: Sequence[Tuple[Path, int]], class_names: Sequence[str], target: Path
) -> None:
    class_detail = []
    counts = defaultdict(int)
    for _, label in samples:
        counts[label] += 1
    for label, class_name in enumerate(class_names):
        class_detail.append(
            {"class_name": class_name, "class_label": label, "images": counts[label]}
        )
    payload = {
        "dataset_root": str(CONFIG["data_root"]),
        "all_class_images": len(samples),
        "class_detail": class_detail,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_transforms(img_size: int) -> Tuple[T.Compose, T.Compose]:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tfms = T.Compose(
        [
            T.Resize(int(img_size * 1.15)),
            T.RandomResizedCrop(img_size, scale=(0.65, 1.0), ratio=(0.8, 1.25)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2, 0.1),
            T.RandomAutocontrast(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    eval_tfms = T.Compose(
        [
            T.Resize(int(img_size * 1.1)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    return train_tfms, eval_tfms


class SceneDataset(Dataset):
    def __init__(self, samples: Sequence[Tuple[Path, int]], transform: T.Compose):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path, label = self.samples[idx]
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        image = self.transform(image)
        return image, label


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        act: bool = True,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(self.pool(x))
        return x * scale


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 2,
        use_se: bool = True,
    ):
        super().__init__()
        hidden_channels = in_channels * expansion
        self.block = nn.Sequential(
            ConvBNAct(in_channels, hidden_channels, 1),
            ConvBNAct(
                hidden_channels,
                hidden_channels,
                3,
                stride=stride,
                groups=hidden_channels,
            ),
            SqueezeExcitation(hidden_channels) if use_se else nn.Identity(),
            ConvBNAct(hidden_channels, out_channels, 1, act=False),
        )
        self.shortcut = (
            ConvBNAct(in_channels, out_channels, 1, stride=stride, act=False)
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + self.shortcut(x))


class SceneNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(3, 48, 3, stride=2), ConvBNAct(48, 64, 3, stride=1)
        )
        self.stage1 = self._make_stage(64, 96, blocks=2, stride=2)
        self.stage2 = self._make_stage(96, 160, blocks=2, stride=2)
        self.stage3 = self._make_stage(160, 224, blocks=3, stride=2)
        self.stage4 = self._make_stage(224, 320, blocks=2, stride=2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(320),
            nn.Dropout(p=0.2),
            nn.Linear(320, num_classes),
        )

    def _make_stage(
        self, in_channels: int, out_channels: int, blocks: int, stride: int
    ) -> nn.Sequential:
        layers: List[nn.Module] = [
            ResidualBlock(in_channels, out_channels, stride=stride)
        ]
        for _ in range(1, blocks):
            layers.append(
                ResidualBlock(out_channels, out_channels, stride=1, use_se=False)
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x)


def get_autocast(device: torch.device):
    if hasattr(torch, "autocast") and device.type in {"cuda", "mps"}:
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return contextlib.nullcontext()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    start = time.time()
    autocast_ctx = get_autocast(device)
    for step, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            outputs = model(images)
            loss = criterion(outputs, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if CONFIG["max_grad_norm"]:
                nn.utils.clip_grad_norm_(model.parameters(), CONFIG["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if CONFIG["max_grad_norm"]:
                nn.utils.clip_grad_norm_(model.parameters(), CONFIG["max_grad_norm"])
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        running_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        running_corrects += (preds == labels).sum().item()
        total += labels.size(0)
        if (step + 1) % 20 == 0:
            print(
                f"Epoch {epoch + 1} Step {step + 1}/{len(loader)} - Loss: {running_loss / total:.4f} Acc: {running_corrects / total:.4f}"
            )
    epoch_loss = running_loss / max(1, total)
    epoch_acc = running_corrects / max(1, total)
    elapsed = time.time() - start
    print(
        f"Epoch {epoch + 1} train done in {elapsed:.1f}s - loss {epoch_loss:.4f} acc {epoch_acc:.4f}"
    )
    return epoch_loss, epoch_acc


def compute_topk(logits: torch.Tensor, labels: torch.Tensor, k: int = 1) -> int:
    _, indices = torch.topk(logits, k=min(k, logits.size(1)), dim=1)
    correct = indices.eq(labels.unsqueeze(1).expand_as(indices))
    return correct.sum().item()


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float, float]:
    model.eval()
    loss_sum = 0.0
    total = 0
    top1 = 0
    top2 = 0
    autocast_ctx = get_autocast(device)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            with autocast_ctx:
                outputs = model(images)
                loss = criterion(outputs, labels)
            loss_sum += loss.item() * labels.size(0)
            total += labels.size(0)
            top1 += compute_topk(outputs, labels, k=1)
            top2 += compute_topk(outputs, labels, k=2)
    loss_avg = loss_sum / max(1, total)
    top1_acc = top1 / max(1, total)
    top2_acc = top2 / max(1, total)
    print(
        f"Epoch {epoch + 1} eval - loss {loss_avg:.4f} top1 {top1_acc:.4f} top2 {top2_acc:.4f}"
    )
    return loss_avg, top1_acc, top2_acc


def preview_predictions(
    model: nn.Module,
    dataset: Dataset,
    indices: Iterable[int],
    device: torch.device,
    class_names: Sequence[str],
) -> None:
    model.eval()
    with torch.no_grad():
        for idx in indices:
            if idx >= len(dataset):
                continue
            image, label = dataset[idx]
            image = image.unsqueeze(0).to(device)
            logits = model(image)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1).item()
            confidence = probs[0, pred].item()
            print(
                f"样本ID：{idx}, 真实标签：{class_names[label]}, 预测值：{class_names[pred]}, 置信度：{confidence:.2f}"
            )


cfg = CONFIG
set_seed(cfg["seed"])
ensure_dataset(cfg["data_root"])
cfg["work_dir"].mkdir(parents=True, exist_ok=True)
cfg["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
samples, class_names = list_image_samples(cfg["data_root"])
train_samples, val_samples = stratified_split(samples, cfg["val_ratio"], cfg["seed"])
export_splits(train_samples, cfg["train_txt"])
export_splits(val_samples, cfg["eval_txt"])
save_readme(samples, class_names, cfg["readme_json"])
train_tfms, eval_tfms = build_transforms(cfg["input_size"])
train_dataset = SceneDataset(train_samples, train_tfms)
val_dataset = SceneDataset(val_samples, eval_tfms)
device = get_device()
print(f"使用设备: {device}")
pin_memory = device.type in {"cuda", "mps"}
train_loader = DataLoader(
    train_dataset,
    batch_size=cfg["batch_size"],
    shuffle=True,
    num_workers=cfg["num_workers"],
    pin_memory=pin_memory,
    drop_last=False,
)
eval_loader = DataLoader(
    val_dataset,
    batch_size=cfg["eval_batch_size"],
    shuffle=False,
    num_workers=max(1, cfg["num_workers"] // 2),
    pin_memory=pin_memory,
)
model = SceneNet(num_classes=len(class_names)).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg["learning_rate"],
    weight_decay=cfg["weight_decay"],
    betas=(0.9, 0.98),
)
steps_per_epoch = max(1, len(train_loader))
total_steps = steps_per_epoch * cfg["num_epochs"]
warmup_steps = max(1, int(total_steps * cfg["warmup_pct"]))
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=cfg["learning_rate"],
    total_steps=total_steps,
    pct_start=warmup_steps / total_steps,
    div_factor=25.0,
    final_div_factor=1e3,
)
scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
best_acc = 0.0
for epoch in range(cfg["num_epochs"]):
    train_one_epoch(
        model, train_loader, criterion, optimizer, scheduler, scaler, device, epoch
    )
    _, val_top1, val_top2 = evaluate(model, eval_loader, criterion, device, epoch)
    if val_top1 > best_acc:
        best_acc = val_top1
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "top1": val_top1,
                "class_names": class_names,
            },
            cfg["best_model_path"],
        )
        print(f"保存最佳模型，Top-1={val_top1:.4f}")
if cfg["best_model_path"].exists():
    checkpoint = torch.load(cfg["best_model_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"最佳模型: epoch {checkpoint['epoch'] + 1}, top1={checkpoint['top1']:.4f}")
preview_indices = [2, 38, 56, 92, 100, 303]
preview_predictions(model, val_dataset, preview_indices, device, class_names)
