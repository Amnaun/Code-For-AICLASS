import gzip
import os
import struct
from typing import Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class FashionMNISTRaw(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        image = self.images[index]
        label = int(self.labels[index])
        pil_image = Image.fromarray(image, mode="L")
        if self.transform:
            image_tensor = self.transform(pil_image)
        else:
            image_tensor = transforms.ToTensor()(pil_image)
        return image_tensor, label

    def __len__(self):
        return len(self.labels)


def _load_idx_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        buf = f.read()
    magic, num, rows, cols = struct.unpack_from(">IIII", buf, 0)
    if magic != 2051:
        raise ValueError(f"Unexpected magic number {magic} in image file {path}")
    np_array = np.frombuffer(buf, dtype=np.uint8, offset=16)
    images = np_array.reshape(num, rows, cols)
    return images


def _load_idx_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        buf = f.read()
    magic, num = struct.unpack_from(">II", buf, 0)
    if magic != 2049:
        raise ValueError(f"Unexpected magic number {magic} in label file {path}")
    np_array = np.frombuffer(buf, dtype=np.uint8, offset=8)
    labels = np_array.reshape(num)
    return labels


def load_raw_dataset(
    raw_dir: str, transform
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    train_images_path = os.path.join(raw_dir, "train-images-idx3-ubyte.gz")
    train_labels_path = os.path.join(raw_dir, "train-labels-idx1-ubyte.gz")
    test_images_path = os.path.join(raw_dir, "t10k-images-idx3-ubyte.gz")
    test_labels_path = os.path.join(raw_dir, "t10k-labels-idx1-ubyte.gz")

    for p in [train_images_path, train_labels_path, test_images_path, test_labels_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing Fashion-MNIST raw file: {p}")

    train_images = _load_idx_images(train_images_path)
    train_labels = _load_idx_labels(train_labels_path)
    test_images = _load_idx_images(test_images_path)
    test_labels = _load_idx_labels(test_labels_path)

    train_set = FashionMNISTRaw(train_images, train_labels, transform=transform)
    test_set = FashionMNISTRaw(test_images, test_labels, transform=transform)
    return train_set, test_set


def get_dataloaders(data_dir: str, batch_size: int):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    try:
        train_set = datasets.FashionMNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
        test_set = datasets.FashionMNIST(
            root=data_dir, train=False, download=True, transform=transform
        )
    except Exception as err:
        print(f"Falling back to raw dataset loader due to: {err}")
        raw_dir = os.getenv(
            "FASHION_MNIST_RAW", os.path.join(data_dir, "FashionMNIST", "raw")
        )
        train_set, test_set = load_raw_dataset(raw_dir, transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, test_set


def build_model() -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )


def train(model: nn.Module, loader: DataLoader, criterion, optimizer, device):
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()


def evaluate(model: nn.Module, loader: DataLoader, device):
    model.eval()
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return {"loss": avg_loss, "acc": accuracy}


def sample_predictions(model: nn.Module, dataset, device):
    model.eval()
    label_list = [
        "t-shirt",
        "trouser",
        "pullover",
        "dress",
        "coat",
        "sandal",
        "shirt",
        "sneaker",
        "bag",
        "ankle boot",
    ]
    indices = [188, 37, 98, 308, 202, 34]

    with torch.no_grad():
        for idx in indices:
            image, label = dataset[idx]
            image_device = image.unsqueeze(0).to(device)
            logits = model(image_device)
            pred = torch.argmax(logits, dim=1).item()
            print(
                f"第{idx}条记录 真实值：{label_list[label]}({label})  预测值：{label_list[pred]}({pred})"
            )


torch.manual_seed(24)

data_dir = os.getenv("FASHION_MNIST_DATA", "./data")
batch_size = 128
epochs = 50
learning_rate = 1e-3

device = select_device()

train_loader, test_loader, test_dataset = get_dataloaders(data_dir, batch_size)

model = build_model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    train(model, train_loader, criterion, optimizer, device)
    metrics = evaluate(model, test_loader, device)
    print(
        f"Epoch {epoch + 1}/{epochs} - test loss: {metrics['loss']:.4f} - accuracy: {metrics['acc']:.4f}"
    )

sample_predictions(model, test_dataset, device)

save_path = "inference_model_v2.pth"
torch.save({"model_state_dict": model.state_dict()}, save_path)
print(f"Model checkpoint saved to {save_path}")
