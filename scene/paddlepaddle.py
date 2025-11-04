import os
import zipfile
import random
import json
import paddle
import sys
import numpy as np
from PIL import Image
from PIL import ImageEnhance
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")  # 忽略 warning

"""
参数配置
"""
train_parameters = {
    "input_size": [3, 224, 224],  # 输入图片的shape
    "class_dim": -1,  # 分类数
    "src_path": "/home/aistudio/data/data34226/scenes.zip",  # 原始数据集路径
    "target_path": "/home/aistudio/data/",  # 要解压的路径
    "train_list_path": "/home/aistudio/data/train.txt",  # train.txt路径
    "eval_list_path": "/home/aistudio/data/eval.txt",  # eval.txt路径
    "readme_path": "/home/aistudio/data/readme.json",  # readme.json路径
    "label_dict": {},  # 标签字典
    "num_epochs": 10,  # 训练轮数
    "train_batch_size": 64,  # 训练时每个批次的大小
    "learning_strategy": {  # 优化函数相关的配置
        "lr": 0.00375  # 超参数学习率
    },
}


def get_data_list(target_path, train_list_path, eval_list_path):
    """
    生成数据列表
    """
    # 存放所有类别的信息
    class_detail = []
    # 获取所有类别保存的文件夹名称
    data_list_path = target_path + "scenes/"
    class_dirs = os.listdir(data_list_path)
    # 总的图像数量
    all_class_images = 0
    # 存放类别标签
    class_label = 0
    # 存放类别数目
    class_dim = 0
    # 存储要写进eval.txt和train.txt中的内容
    trainer_list = []
    eval_list = []
    # 读取每个类别，['river', 'lawn','church','ice','desert']
    for class_dir in class_dirs:
        if class_dir != ".DS_Store":
            class_dim += 1
            # 每个类别的信息
            class_detail_list = {}
            eval_sum = 0
            trainer_sum = 0
            # 统计每个类别有多少张图片
            class_sum = 0
            # 获取类别路径
            path = data_list_path + class_dir
            # 获取所有图片
            img_paths = os.listdir(path)
            for img_path in img_paths:  # 遍历文件夹下的每个图片
                name_path = path + "/" + img_path  # 每张图片的路径
                if class_sum % 8 == 0:  # 每8张图片取一个做验证数据
                    eval_sum += 1  # test_sum为测试数据的数目
                    eval_list.append(name_path + "\t%d" % class_label + "\n")
                else:
                    trainer_sum += 1
                    trainer_list.append(
                        name_path + "\t%d" % class_label + "\n"
                    )  # trainer_sum测试数据的数目
                class_sum += 1  # 每类图片的数目
                all_class_images += 1  # 所有类图片的数目

            # 说明的json文件的class_detail数据
            class_detail_list["class_name"] = class_dir  # 类别名称
            class_detail_list["class_label"] = class_label  # 类别标签
            class_detail_list["class_eval_images"] = eval_sum  # 该类数据的测试集数目
            class_detail_list["class_trainer_images"] = (
                trainer_sum  # 该类数据的训练集数目
            )
            class_detail.append(class_detail_list)
            # 初始化标签列表
            train_parameters["label_dict"][str(class_label)] = class_dir
            class_label += 1

    # 初始化分类数
    train_parameters["class_dim"] = class_dim

    # 乱序
    random.shuffle(eval_list)
    with open(eval_list_path, "a") as f:
        for eval_image in eval_list:
            f.write(eval_image)

    random.shuffle(trainer_list)
    with open(train_list_path, "a") as f2:
        for train_image in trainer_list:
            f2.write(train_image)

    # 说明的json文件信息
    readjson = {}
    readjson["all_class_name"] = data_list_path  # 文件父目录
    readjson["all_class_images"] = all_class_images
    readjson["class_detail"] = class_detail
    jsons = json.dumps(readjson, sort_keys=True, indent=4, separators=(",", ": "))
    with open(train_parameters["readme_path"], "w") as f:
        f.write(jsons)
    print("生成数据列表完成！")


"""
参数初始化
"""
src_path = train_parameters["src_path"]
target_path = train_parameters["target_path"]
train_list_path = train_parameters["train_list_path"]
eval_list_path = train_parameters["eval_list_path"]
batch_size = train_parameters["train_batch_size"]

"""
解压原始数据到指定路径
"""
unzip_data(src_path, target_path)

"""
划分训练集与验证集，乱序，生成数据列表
"""
# 每次生成数据列表前，首先清空train.txt和eval.txt
with open(train_list_path, "w") as f:
    f.seek(0)
    f.truncate()
with open(eval_list_path, "w") as f:
    f.seek(0)
    f.truncate()

# 生成数据列表
get_data_list(target_path, train_list_path, eval_list_path)

import paddle
import paddle.vision.transforms as T
import numpy as np
from PIL import Image


class SceneDataset(paddle.io.Dataset):
    """
    5类场景数据集类的定义
    """

    def __init__(self, mode="train"):
        """
        初始化函数
        """
        assert mode in ["train", "eval"], "mode is one of train, eval."

        self.data = []

        with open("data/{}.txt".format(mode)) as f:
            for line in f.readlines():
                info = line.strip().split("\t")

                if len(info) > 0:
                    self.data.append([info[0].strip(), info[1].strip()])

        self.transforms = T.Compose(
            [
                T.Resize((224, 224)),  # 图像大小修改
                T.ToTensor(),  # 数据的格式转换和标准化 HWC => CHW
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __getitem__(self, index):
        """
        根据索引获取单个样本
        """
        image_file, label = self.data[index]
        image = Image.open(image_file)

        if image.mode != "RGB":
            image = image.convert("RGB")

        image = self.transforms(image)

        return image, np.array(label, dtype="int64")

    def __len__(self):
        """
        获取样本总数
        """
        return len(self.data)


"""
构造数据提供器
"""
train_dataset = SceneDataset(mode="train")
test_dataset = SceneDataset(mode="eval")

# 定义CNN网络
import paddle
import paddle.nn.functional as F
import paddle.nn as nn


class ConvPool(nn.Layer):
    """卷积+池化"""

    def __init__(
        self,
        num_channels,
        num_filters,
        filter_size,
        pool_size,
        pool_stride,
        groups,
        conv_stride=1,
        conv_padding=1,
    ):
        super(ConvPool, self).__init__()

        self._conv2d_list = []

        for i in range(groups):
            # add_sublayer 添加子层实例。可以通过self.name访问该sublayer
            conv2d = self.add_sublayer(  # 返回一个由所有子层组成的列表。
                "bb_%d" % i,
                paddle.nn.Conv2D(
                    in_channels=num_channels,  # 通道数
                    out_channels=num_filters,  # 卷积核个数
                    kernel_size=filter_size,  # 卷积核大小
                    stride=conv_stride,  # 步长
                    padding=conv_padding,
                ),  # padding大小，默认为0)
            )
            num_channels = num_filters
            self._conv2d_list.append(conv2d)

        self._pool2d = paddle.nn.MaxPool2D(
            kernel_size=pool_size,  # 池化核大小
            stride=pool_stride,  # 池化步长
        )

    def forward(self, inputs):
        x = inputs
        for conv in self._conv2d_list:
            x = conv(x)
            x = F.relu(x)
        x = self._pool2d(x)
        return x


class VGGNet(nn.Layer):
    """
    VGG网络
    """

    def __init__(self):
        super(VGGNet, self).__init__()
        self.convpool01 = ConvPool(
            3, 64, 3, 2, 2, 2
        )  # 3:通道数，64：卷积核个数，3:卷积核大小，2:池化核大小，2:池化步长，2:连续卷积个数
        self.convpool02 = ConvPool(64, 128, 3, 2, 2, 2)
        self.convpool03 = ConvPool(128, 256, 3, 2, 2, 3)
        self.convpool04 = ConvPool(256, 512, 3, 2, 2, 3)
        self.convpool05 = ConvPool(512, 512, 3, 2, 2, 3)
        self.pool_5_shape = 512 * 7 * 7
        # self.flatten = paddle.nn.Flatten()
        self.fc01 = nn.Linear(in_features=self.pool_5_shape, out_features=4096)
        self.fc02 = nn.Linear(in_features=4096, out_features=4096)
        self.fc03 = nn.Linear(
            in_features=4096, out_features=train_parameters["class_dim"]
        )

    # def forward(self, inputs, label=None):
    def forward(self, inputs):
        # print('first'+ 50*'*')
        # print(inputs.shape) #[8, 3, 224, 224]
        """前向计算"""
        out = self.convpool01(inputs)
        # print(out.shape)           #[8, 64, 112, 112]
        # out = F.relu(out)
        out = self.convpool02(out)
        # print(out.shape)           #[8, 128, 56, 56]
        # out = F.relu(out)
        out = self.convpool03(out)
        # print(out.shape)           #[8, 256, 28, 28]
        # out = F.relu(out)
        out = self.convpool04(out)
        # print(out.shape)           #[8, 512, 14, 14]
        # out = F.relu(out)
        out = self.convpool05(out)
        # print(out.shape)           #[8, 512, 7, 7]
        # out = F.relu(out)

        out = paddle.reshape(out, shape=[-1, 512 * 7 * 7])
        out = self.fc01(out)
        out = F.relu(out)
        out = self.fc02(out)
        out = F.relu(out)
        out = self.fc03(out)
        out = F.softmax(out)
        # out = paddle.flatten(out)
        # if label is not None:
        #     print('label:', label)
        #     acc = paddle.metric.accuracy(input=out, label=label)
        #     return out, acc
        # else:
        return out


import paddle
from paddle import Model

vgg = VGGNet()
model = Model(vgg)
model.summary((1, 3, 224, 224))


def create_optim(parameters):
    step_each_epoch = int(3498 // 64)
    lr = paddle.optimizer.lr.CosineAnnealingDecay(
        learning_rate=0.00375, T_max=step_each_epoch * 10
    )

    return paddle.optimizer.Momentum(
        learning_rate=lr,
        parameters=parameters,
        weight_decay=paddle.regularizer.L2Decay(0.000001),
    )


# 模型训练配置
model.prepare(
    create_optim(model.parameters()),  # 优化器
    paddle.nn.CrossEntropyLoss(),  # 损失函数
    paddle.metric.Accuracy(topk=(1, 2)),
)  # 评估指标

# 训练可视化VisualDL工具的回调函数
visualdl = paddle.callbacks.VisualDL(log_dir="visualdl_log")
# 启动模型全流程训练
model.fit(
    train_dataset,  # 训练数据集
    test_dataset,  # 评估数据集
    epochs=10,  # 总的训练轮次
    batch_size=64,  # 批次计算的样本量大小
    shuffle=True,  # 是否打乱样本集
    verbose=1,  # 日志展示格式
    save_dir="./chk_points/",  # 分阶段的训练模型存储路径
    callbacks=[visualdl],
)  # 回调函数使用

model.save("model_save_dir")

print("测试数据集样本量：{}".format(len(test_dataset)))
# 执行预测
result = model.predict(test_dataset)

# 样本映射
LABEL_MAP = ["ice", "river", "lawn", "church", "desert"]

# 随机取样本展示
indexs = [2, 38, 56, 92, 100, 303]

for idx in indexs:
    predict_label = np.argmax(result[0][idx])
    real_label = test_dataset.__getitem__(idx)[1]
    print(real_label)
    print(
        "样本ID：{}, 真实标签：{}, 预测值：{}".format(
            idx, LABEL_MAP[real_label], LABEL_MAP[predict_label]
        )
    )
model.save(".", training=False)
