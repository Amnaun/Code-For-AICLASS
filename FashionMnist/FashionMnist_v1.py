import gzip
import os
import time
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import paddle
import paddle.nn as nn
from paddle.io import Dataset

import os
import gzip
import struct
import numpy as np
from PIL import Image
import paddle
from paddle.io import Dataset


class FashionMNIST(Dataset):
    """
    步骤一：继承paddle.io.Dataset类
    """

    def __init__(self, path="./", mode="train", transform=None):
        """
        步骤二：实现构造函数，定义数据读取方式，划分训练和测试数据集
        """
        backend = paddle.vision.get_image_backend()
        if backend not in ["pil", "cv2"]:
            raise ValueError(
                "Expected backend are one of ['pil', 'cv2'], but got {}".format(backend)
            )
        self.backend = backend
        self.mode = mode.lower()

        self.images_data_path = os.path.join(path, "%s-images-idx3-ubyte.gz" % mode)
        self.labels_data_path = os.path.join(path, "%s-labels-idx1-ubyte.gz" % mode)
        self.transform = transform
        # read dataset into memory
        self._parse_dataset()
        self.dtype = paddle.get_default_dtype()

    def _parse_dataset(self, buffer_size=100):
        self.images = []
        self.labels = []
        with gzip.GzipFile(self.images_data_path, "rb") as image_file:
            img_buf = image_file.read()
            with gzip.GzipFile(self.labels_data_path, "rb") as label_file:
                lab_buf = label_file.read()

                step_label = 0
                offset_img = 0
                # read from Big-endian
                # get file info from magic byte
                # image file : 16B
                magic_byte_img = ">IIII"
                magic_img, image_num, rows, cols = struct.unpack_from(
                    magic_byte_img, img_buf, offset_img
                )
                offset_img += struct.calcsize(magic_byte_img)

                offset_lab = 0
                # label file : 8B
                magic_byte_lab = ">II"
                magic_lab, label_num = struct.unpack_from(
                    magic_byte_lab, lab_buf, offset_lab
                )
                offset_lab += struct.calcsize(magic_byte_lab)

                while True:
                    if step_label >= label_num:
                        break
                    fmt_label = ">" + str(buffer_size) + "B"
                    labels = struct.unpack_from(fmt_label, lab_buf, offset_lab)
                    offset_lab += struct.calcsize(fmt_label)
                    step_label += buffer_size

                    fmt_images = ">" + str(buffer_size * rows * cols) + "B"
                    images_temp = struct.unpack_from(fmt_images, img_buf, offset_img)
                    images = np.reshape(images_temp, (buffer_size, rows * cols)).astype(
                        "float32"
                    )
                    offset_img += struct.calcsize(fmt_images)

                    for i in range(buffer_size):
                        self.images.append(images[i, :])
                        self.labels.append(np.array([labels[i]]).astype("int64"))

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        image = np.reshape(image, [28, 28])

        if self.backend == "pil":
            image = Image.fromarray(image.astype("uint8"), mode="L")

        if self.transform is not None:
            image = self.transform(image)

        if self.backend == "pil":
            return image, label.astype("int64")

        return image.astype(self.dtype), label.astype("int64")

    def __len__(self):
        return len(self.labels)


# 实例化数据集
import paddle.vision.transforms as T

transform = T.Compose(
    [T.Normalize(mean=[127.5], std=[127.5], data_format="CHW"), T.ToTensor()]
)

fashion_mnist_train_dataset = FashionMNIST(
    path="data/data7688", mode="train", transform=transform
)
fashion_mnist_test_dataset = FashionMNIST(
    path="data/data7688", mode="t10k", transform=transform
)

from paddle.nn import Linear
import paddle.nn.functional as F
from paddle.nn import Layer


class MultilayerPerceptron(Layer):
    def __init__(self):
        super(MultilayerPerceptron, self).__init__()
        self.linear1 = Linear(in_features=1 * 28 * 28, out_features=100)
        self.linear2 = Linear(in_features=100, out_features=100)
        self.linear3 = Linear(in_features=100, out_features=10)

    def forward(self, inputs):
        x = paddle.flatten(inputs, start_axis=1, stop_axis=-1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        y = F.softmax(x)
        return y


# 模型网络结构搭建
network = paddle.nn.Sequential(
    paddle.nn.Flatten(),  # 拉平，将 (28, 28) => (784)
    paddle.nn.Linear(784, 100),  # 隐层：线性变换层
    paddle.nn.ReLU(),  # 激活函数
    paddle.nn.Linear(100, 100),  # 输出层
    paddle.nn.ReLU(),  # 激活函数
    paddle.nn.Linear(100, 10),  # 输出层
    paddle.nn.Softmax(),  # 激活函数
)

import paddle
from paddle import Model

# model= Model(MultilayerPerceptron())
model = Model(network)

# 配置模型
model.prepare(
    paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters()),
    paddle.nn.CrossEntropyLoss(),
    paddle.metric.Accuracy(),
)

model.summary((1, 28, 28))

eval_result = model.evaluate(fashion_mnist_test_dataset, verbose=1)

print(eval_result)

# 进行预测操作
predict_result = model.predict(fashion_mnist_test_dataset)

# 定义画图方法
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

# 抽样展示
indexs = [2, 15, 38, 211, 222, 323]

for idx in indexs:
    print(
        "第{}条记录 真实值： {}   预测值：{}".format(
            idx,
            fashion_mnist_test_dataset[idx][1][0],
            np.argmax(predict_result[0][idx]),
        )
    )
model.save("inference_model", training=False)  # save for inference
