import os
import zipfile
import random
import json
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

train_parameters = {
    "input_size": [1, 20, 20],
    "class_dim": -1,
    "src_path": "data/characterData.zip",
    "target_path": "dataset/characterData",
    "train_list_path": "./train_data.txt",
    "eval_list_path": "./val_data.txt",
    "label_dict": {},
    "readme_path": "readme.json",
    "num_epochs": 50,
    "train_batch_size": 32,
    "learning_strategy": {"lr": 0.001},
}

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"正在使用设备: {device}")


def get_data_list(target_path, train_list_path, eval_list_path):
    class_detail = []
    data_list_path = target_path
    class_dirs = os.listdir(data_list_path)
    if "__MACOSX" in class_dirs:
        class_dirs.remove("__MACOSX")
    all_class_images = 0
    class_label = 0
    class_dim = 0
    trainer_list = []
    eval_list = []
    for class_dir in class_dirs:
        if class_dir != ".DS_Store" and os.path.isdir(
            os.path.join(data_list_path, class_dir)
        ):
            class_dim += 1
            class_detail_list = {}
            eval_sum = 0
            trainer_sum = 0
            class_sum = 0
            path = os.path.join(data_list_path, class_dir)
            img_paths = os.listdir(path)
            for img_path in img_paths:
                if img_path == ".DS_Store":
                    continue
                name_path = os.path.join(path, img_path)
                if class_sum % 10 == 0:
                    eval_sum += 1
                    eval_list.append(name_path + "\t%d" % class_label + "\n")
                else:
                    trainer_sum += 1
                    trainer_list.append(name_path + "\t%d" % class_label + "\n")
                class_sum += 1
                all_class_images += 1
            class_detail_list["class_name"] = class_dir
            class_detail_list["class_label"] = class_label
            class_detail_list["class_eval_images"] = eval_sum
            class_detail_list["class_trainer_images"] = trainer_sum
            class_detail.append(class_detail_list)
            train_parameters["label_dict"][str(class_label)] = class_dir
            class_label += 1
    train_parameters["class_dim"] = class_dim
    print(f"动态检测到 {class_dim} 个类别。")
    random.shuffle(eval_list)
    with open(eval_list_path, "a") as f:
        for eval_image in eval_list:
            f.write(eval_image)
    random.shuffle(trainer_list)
    with open(train_list_path, "a") as f2:
        for train_image in trainer_list:
            f2.write(train_image)
    readjson = {}
    readjson["all_class_name"] = data_list_path
    readjson["all_class_images"] = all_class_images
    readjson["class_detail"] = class_detail
    jsons = json.dumps(readjson, sort_keys=True, indent=4, separators=(",", ": "))
    with open(train_parameters["readme_path"], "w") as f:
        f.write(jsons)
    return True


class CharacterDataset(Dataset):
    def __init__(self, list_path, transform=None):
        super(CharacterDataset, self).__init__()
        self.file_list = []
        with open(list_path, "r") as f:
            for line in f:
                img_path, label = line.strip().split("\t")
                self.file_list.append((img_path, int(label)))
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path, label = self.file_list[idx]
        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, label


class MyCNN(nn.Module):
    def __init__(self, num_classes):
        super(MyCNN, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.fc_stack = nn.Sequential(
            nn.Linear(32 * 5 * 5, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        logits = self.fc_stack(x)
        return logits


Batchs = []
all_train_accs = []


def draw_train_acc(Batchs, train_accs):
    plt.figure()
    title = "training accs (PyTorch+MPS+CNN_v2)"
    plt.title(title, fontsize=24)
    plt.xlabel("batch", fontsize=14)
    plt.ylabel("acc", fontsize=14)
    plt.plot(Batchs, train_accs, color="green", label="training accs")
    plt.legend()
    plt.grid()
    plt.savefig("train_acc.png")
    print("训练准确率曲线图已保存为 train_acc.png")


all_train_loss = []


def draw_train_loss(Batchs, train_loss):
    plt.figure()
    title = "training loss"
    plt.title(title, fontsize=24)
    plt.xlabel("batch", fontsize=14)
    plt.ylabel("loss", fontsize=14)
    plt.plot(Batchs, train_loss, color="red", label="training loss")
    plt.legend()
    plt.grid()
    plt.savefig("train_loss.png")
    print("训练损失曲线图已保存为 train_loss.png")


target_path = train_parameters["target_path"]
train_list_path = train_parameters["train_list_path"]
eval_list_path = train_parameters["eval_list_path"]
batch_size = train_parameters["train_batch_size"]
with open(train_list_path, "w") as f:
    f.seek(0)
    f.truncate()
with open(eval_list_path, "w") as f:
    f.seek(0)
    f.truncate()
if not get_data_list(target_path, train_list_path, eval_list_path):
    print("数据列表生成失败，请检查错误信息并核对 'target_path'。")
    import sys

    sys.exit()
num_classes = train_parameters["class_dim"]
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = CharacterDataset(train_list_path, transform=transform)
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
)
eval_dataset = CharacterDataset(eval_list_path, transform=transform)
eval_loader = DataLoader(
    eval_dataset, batch_size=batch_size, shuffle=False, drop_last=True
)
model = MyCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(), lr=train_parameters["learning_strategy"]["lr"]
)
epochs_num = train_parameters["num_epochs"]
Batch = 0
print("\n--- 开始训练 ---")
for pass_num in range(epochs_num):
    model.train()
    for batch_id, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        predicts = model(images)
        loss = criterion(predicts, labels)
        loss.backward()
        optimizer.step()
        _, predicted_labels = torch.max(predicts.data, 1)
        acc = (predicted_labels == labels).float().mean()
        if batch_id != 0 and batch_id % 50 == 0:
            Batch = Batch + 50
            Batchs.append(Batch)
            all_train_loss.append(loss.item())
            all_train_accs.append(acc.item())
            print(
                f"train_pass:{pass_num}, batch_id:{batch_id}, "
                f"train_loss:{loss.item():.6f}, train_acc:{acc.item():.6f}"
            )
model_save_path = "MyCNN.pth"
torch.save(model.state_dict(), model_save_path)
print(f"模型已保存到 {model_save_path}")
draw_train_acc(Batchs, all_train_accs)
draw_train_loss(Batchs, all_train_loss)
print("\n--- 开始评估 ---")
accs = []
model.eval()
with torch.no_grad():
    for batch_id, (images, labels) in enumerate(eval_loader):
        images = images.to(device)
        labels = labels.to(device)
        predicts = model(images)
        _, predicted_labels = torch.max(predicts.data, 1)
        acc = (predicted_labels == labels).float().mean()
        accs.append(acc.item())
avg_acc = np.mean(accs)
print(f"在评估集上的平均准确率: {avg_acc:.6f}")
print("\n--- 开始处理车牌并预测 ---")
license_plate_path = "work/车牌.png"
license_plate = cv2.imread(license_plate_path)
if license_plate is None:
    print(f"错误：'{license_plate_path}' 未找到，跳过预测。")
    print("请确保 'work' 文件夹存在，并且 '车牌.png' 在里面。")
else:
    print(f"车牌图像尺寸: {license_plate.shape}")
    gray_plate = cv2.cvtColor(license_plate, cv2.COLOR_RGB2GRAY)
    ret, binary_plate = cv2.threshold(gray_plate, 175, 255, cv2.THRESH_BINARY)
    result = []
    for col in range(binary_plate.shape[1]):
        result.append(0)
        for row in range(binary_plate.shape[0]):
            result[col] = result[col] + binary_plate[row][col] / 255
    character_dict = {}
    num = 0
    i = 0
    while i < len(result):
        if result[i] == 0:
            i += 1
        else:
            index = i + 1
            if index >= len(result):
                break
            while result[index] != 0:
                index += 1
                if index >= len(result):
                    break
            character_dict[num] = [i, index - 1]
            num += 1
            i = index
    characters = []
    if not os.path.exists("work"):
        os.makedirs("work")
    for i in range(8):
        if i == 2:
            continue
        if i not in character_dict:
            print(f"警告: 无法分割出第 {i} 个字符。")
            continue
        char_width = character_dict[i][1] - character_dict[i][0]
        if char_width <= 0:
            print(f"警告: 第 {i} 个字符宽度为0，跳过。")
            continue
        padding = (170 - char_width) / 2
        padding = max(0, int(padding))
        ndarray = np.pad(
            binary_plate[:, character_dict[i][0] : character_dict[i][1]],
            ((0, 0), (padding, padding)),
            "constant",
            constant_values=(0, 0),
        )
        ndarray = cv2.resize(ndarray, (20, 20))
        save_path = f"work/{i}.png"
        cv2.imwrite(save_path, ndarray)
        print(f"已保存分割字符: {save_path}")
        characters.append(ndarray)

    def load_image_pytorch(path, device):
        img = Image.open(path).convert("L")
        transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = transform(img)
        return img_tensor.unsqueeze(0).to(device)

    print("Label Dict:", train_parameters["label_dict"])
    match = {
        "A": "A",
        "B": "B",
        "C": "C",
        "D": "D",
        "E": "E",
        "F": "F",
        "G": "G",
        "H": "H",
        "I": "I",
        "J": "J",
        "K": "K",
        "L": "L",
        "M": "M",
        "N": "N",
        "O": "O",
        "P": "P",
        "Q": "Q",
        "R": "R",
        "S": "S",
        "T": "T",
        "U": "U",
        "V": "V",
        "W": "W",
        "X": "X",
        "Y": "Y",
        "Z": "Z",
        "yun": "云",
        "cuan": "川",
        "hei": "黑",
        "zhe": "浙",
        "ning": "宁",
        "jin": "津",
        "gan": "赣",
        "hu": "沪",
        "liao": "辽",
        "jl": "吉",
        "qing": "青",
        "zang": "藏",
        "e1": "鄂",
        "meng": "蒙",
        "gan1": "甘",
        "qiong": "琼",
        "shan": "陕",
        "min": "闽",
        "su": "苏",
        "xin": "新",
        "wan": "皖",
        "jing": "京",
        "xiang": "湘",
        "gui": "贵",
        "yu1": "渝",
        "yu": "豫",
        "ji": "冀",
        "yue": "粤",
        "gui1": "桂",
        "sx": "晋",
        "lu": "鲁",
        "0": "0",
        "1": "1",
        "2": "2",
        "3": "3",
        "4": "4",
        "5": "5",
        "6": "6",
        "7": "7",
        "8": "8",
        "9": "9",
    }
    L = 0
    LABEL = {}
    for V in train_parameters["label_dict"].values():
        if V in match:
            LABEL[str(L)] = match[V]
        else:
            LABEL[str(L)] = "?"
        L += 1
    model_infer = MyCNN(num_classes).to(device)
    model_infer.load_state_dict(torch.load(model_save_path))
    model_infer.eval()
    lab = []
    with torch.no_grad():
        for i in range(8):
            if i == 2:
                continue
            img_path = f"work/{i}.png"
            if not os.path.exists(img_path):
                continue
            infer_img_tensor = load_image_pytorch(img_path, device)
            result = model_infer(infer_img_tensor)
            pred_label = result.argmax(dim=1).item()
            lab.append(pred_label)
    print(f"预测的标签索引: {lab}")
    try:
        plt.figure()
        plt.imshow(Image.open(license_plate_path))
        plt.axis("off")
        plt.savefig("license_plate_result.png")
        print("车牌识别结果图已保存为 license_plate_result.png")
    except Exception as e:
        print(f"显示或保存车牌图片时出错: {e}")
    print("\n--- 最终预测结果 ---")
    result_str = ""
    for i in range(len(lab)):
        if str(lab[i]) in LABEL:
            result_str += LABEL[str(lab[i])]
        else:
            result_str += "?"
    print(result_str)
