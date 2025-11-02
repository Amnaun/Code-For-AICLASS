# 导入需要的包
import os
import zipfile
import random
import json
import cv2
import numpy as np
from PIL import Image
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear
import matplotlib.pyplot as plt


"""
参数配置
"""
train_parameters = {
    "input_size": [1, 20, 20],  # 输入图片的shape
    "class_dim": -1,  # 分类数
    "src_path": "data/data23617/characterData.zip",  # 原始数据集路径
    "target_path": "/home/aistudio/data/dataset",  # 要解压的路径
    "train_list_path": "./train_data.txt",  # train_data.txt路径
    "eval_list_path": "./val_data.txt",  # eval_data.txt路径
    "label_dict": {},  # 标签字典
    "readme_path": "/home/aistudio/data/readme.json",  # readme.json路径
    "num_epochs": 50,  # 训练轮数
    "train_batch_size": 32,  # 批次的大小
    "learning_strategy": {  # 优化函数相关的配置
        "lr": 0.001  # 超参数学习率
    },
}


def unzip_data(src_path, target_path):
    """
    解压原始数据集，将src_path路径下的zip包解压至data/dataset目录下
    """

    if not os.path.isdir(target_path):
        z = zipfile.ZipFile(src_path, "r")
        z.extractall(path=target_path)
        z.close()
    else:
        print("文件已解压")


def get_data_list(target_path, train_list_path, eval_list_path):
    """
    生成数据列表
    """
    # 存放所有类别的信息
    class_detail = []
    # 获取所有类别保存的文件夹名称
    data_list_path = target_path
    class_dirs = os.listdir(data_list_path)
    if "__MACOSX" in class_dirs:
        class_dirs.remove("__MACOSX")
    # #总的图像数量
    all_class_images = 0
    # #存放类别标签
    class_label = 0
    # #存放类别数目
    class_dim = 0
    # #存储要写进eval.txt和train.txt中的内容
    trainer_list = []
    eval_list = []
    # 读取每个类别
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
            path = os.path.join(data_list_path, class_dir)
            # print(path)
            # 获取所有图片
            img_paths = os.listdir(path)
            for img_path in img_paths:  # 遍历文件夹下的每个图片
                if img_path == ".DS_Store":
                    continue
                name_path = os.path.join(path, img_path)  # 每张图片的路径
                if class_sum % 10 == 0:  # 每10张图片取一个做验证数据
                    eval_sum += 1  # eval_sum为测试数据的数目
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
    print(train_parameters)
    # 乱序
    random.shuffle(eval_list)
    with open(eval_list_path, "a") as f:
        for eval_image in eval_list:
            f.write(eval_image)
    # 乱序
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


def data_reader(file_list):
    """
    自定义data_reader
    """

    def reader():
        with open(file_list, "r") as f:
            lines = [line.strip() for line in f]
            for line in lines:
                img_path, lab = line.strip().split("\t")
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.array(img).astype("float32")
                img = img / 255.0
                yield img, int(lab)

    return reader


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

# 每次生成数据列表前，首先清空train.txt和eval.txt
with open(train_list_path, "w") as f:
    f.seek(0)
    f.truncate()
with open(eval_list_path, "w") as f:
    f.seek(0)
    f.truncate()

# 生成数据列表
get_data_list(target_path, train_list_path, eval_list_path)

"""
构造数据提供器
"""
train_reader = paddle.batch(
    data_reader(train_list_path), batch_size=batch_size, drop_last=True
)
eval_reader = paddle.batch(
    data_reader(eval_list_path), batch_size=batch_size, drop_last=True
)

Batch = 0
Batchs = []
all_train_accs = []


def draw_train_acc(Batchs, train_accs):
    title = "training accs"
    plt.title(title, fontsize=24)
    plt.xlabel("batch", fontsize=14)
    plt.ylabel("acc", fontsize=14)
    plt.plot(Batchs, train_accs, color="green", label="training accs")
    plt.legend()
    plt.grid()
    plt.show()


all_train_loss = []


def draw_train_loss(Batchs, train_loss):
    title = "training loss"
    plt.title(title, fontsize=24)
    plt.xlabel("batch", fontsize=14)
    plt.ylabel("loss", fontsize=14)
    plt.plot(Batchs, train_loss, color="red", label="training loss")
    plt.legend()
    plt.grid()
    plt.show()


# 定义DNN网络
class MyDNN(fluid.dygraph.Layer):
    def __init__(self):
        super(MyDNN, self).__init__()
        self.hidden1 = Linear(20 * 20, 200, act="relu")
        self.hidden2 = Linear(200, 100, act="relu")
        self.hidden3 = Linear(100, 100, act="relu")
        self.out = Linear(100, 65, act="softmax")

    def forward(self, input):  # forward 定义执行实际运行时网络的执行逻辑
        x = fluid.layers.reshape(
            input, shape=[-1, 20 * 20]
        )  # -1 表示这个维度的值是从x的元素总数和剩余维度推断出来的，有且只能有一个维度设置为-1
        # print(x.shape)
        x = self.hidden1(x)
        # print('1', x.shape)
        x = self.hidden2(x)
        # print('2',x.shape)
        x = self.hidden3(x)
        # print('3',x.shape)
        y = self.out(x)
        # print('4',y.shape)
        return y


with fluid.dygraph.guard():
    model = MyDNN()  # 模型实例化
    model.train()  # 训练模式
    opt = fluid.optimizer.SGDOptimizer(
        learning_rate=train_parameters["learning_strategy"]["lr"],
        parameter_list=model.parameters(),
    )  # 优化器选用SGD随机梯度下降，学习率为0.001.
    epochs_num = train_parameters["num_epochs"]  # 迭代次数

    for pass_num in range(epochs_num):
        for batch_id, data in enumerate(train_reader()):
            images = np.array([x[0].reshape(1, 20, 20) for x in data], np.float32)
            labels = np.array([x[1] for x in data]).astype("int64")
            labels = labels[:, np.newaxis]
            image = fluid.dygraph.to_variable(images)
            label = fluid.dygraph.to_variable(labels)

            predict = model(image)  # 数据传入model

            loss = fluid.layers.cross_entropy(predict, label)
            avg_loss = fluid.layers.mean(loss)  # 获取loss值

            acc = fluid.layers.accuracy(predict, label)  # 计算精度

            if batch_id != 0 and batch_id % 50 == 0:
                Batch = Batch + 50
                Batchs.append(Batch)
                all_train_loss.append(avg_loss.numpy()[0])
                all_train_accs.append(acc.numpy()[0])

                print(
                    "train_pass:{},batch_id:{},train_loss:{},train_acc:{}".format(
                        pass_num, batch_id, avg_loss.numpy(), acc.numpy()
                    )
                )

            avg_loss.backward()
            opt.minimize(avg_loss)  # 优化器对象的minimize方法对参数进行更新
            model.clear_gradients()  # model.clear_gradients()来重置梯度
    fluid.save_dygraph(model.state_dict(), "MyDNN")  # 保存模型

draw_train_acc(Batchs, all_train_accs)
draw_train_loss(Batchs, all_train_loss)


# 模型评估
with fluid.dygraph.guard():
    accs = []
    model_dict, _ = fluid.load_dygraph("MyDNN")
    model = MyDNN()
    model.load_dict(model_dict)  # 加载模型参数
    model.eval()  # 训练模式
    for batch_id, data in enumerate(eval_reader()):  # 测试集
        images = np.array([x[0].reshape(1, 20, 20) for x in data], np.float32)
        labels = np.array([x[1] for x in data]).astype("int64")
        labels = labels[:, np.newaxis]
        image = fluid.dygraph.to_variable(images)
        label = fluid.dygraph.to_variable(labels)
        predict = model(image)
        acc = fluid.layers.accuracy(predict, label)
        accs.append(acc.numpy()[0])
        avg_acc = np.mean(accs)
    print(avg_acc)


# 对车牌图片进行处理，分割出车牌中的每一个字符并保存
license_plate = cv2.imread("work/车牌.png")
print(license_plate.shape)
gray_plate = cv2.cvtColor(license_plate, cv2.COLOR_RGB2GRAY)
ret, binary_plate = cv2.threshold(
    gray_plate, 175, 255, cv2.THRESH_BINARY
)  # ret：阈值，binary_plate：根据阈值处理后的图像数据
# 按列统计像素分布
result = []
for col in range(binary_plate.shape[1]):
    result.append(0)
    for row in range(binary_plate.shape[0]):
        result[col] = result[col] + binary_plate[row][col] / 255
# print(result)
# 记录车牌中字符的位置
character_dict = {}
num = 0
i = 0
while i < len(result):
    if result[i] == 0:
        i += 1
    else:
        index = i + 1
        while result[index] != 0:
            index += 1
        character_dict[num] = [i, index - 1]
        num += 1
        i = index
# print(character_dict)
# 将每个字符填充，并存储
characters = []
for i in range(8):
    if i == 2:
        continue
    padding = (170 - (character_dict[i][1] - character_dict[i][0])) / 2
    # 将单个字符图像填充为170*170
    ndarray = np.pad(
        binary_plate[:, character_dict[i][0] : character_dict[i][1]],
        ((0, 0), (int(padding), int(padding))),
        "constant",
        constant_values=(0, 0),
    )
    ndarray = cv2.resize(ndarray, (20, 20))
    cv2.imwrite("work/" + str(i) + ".png", ndarray)
    characters.append(ndarray)


def load_image(path):
    img = paddle.dataset.image.load_image(file=path, is_color=False)
    img = img.astype("float32")
    img = img[np.newaxis,] / 255.0
    return img


# 将标签进行转换
print("Label:", train_parameters["label_dict"])
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
    LABEL[str(L)] = match[V]
    L += 1
print(LABEL)


# 构建预测动态图过程
with fluid.dygraph.guard():
    model = MyDNN()  # 模型实例化
    model_dict, _ = fluid.load_dygraph("MyDNN")
    model.load_dict(model_dict)  # 加载模型参数
    model.eval()  # 评估模式
    lab = []
    for i in range(8):
        if i == 2:
            continue
        infer_imgs = []
        infer_imgs.append(load_image("work/" + str(i) + ".png"))
        infer_imgs = np.array(infer_imgs)
        infer_imgs = fluid.dygraph.to_variable(infer_imgs)
        result = model(infer_imgs)
        lab.append(np.argmax(result.numpy()))
print(lab)
display(Image.open("work/车牌.png"))
for i in range(len(lab)):
    print(LABEL[str(lab[i])], end="")
