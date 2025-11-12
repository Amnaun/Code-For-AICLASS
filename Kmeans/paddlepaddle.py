import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets

# 直接从sklearn中获取数据集
iris = datasets.load_iris()
X = iris.data[:, :4]  # 表示我们取特征空间中的4个维度
print(X.shape)

# 取前两个维度（萼片长度、萼片宽度），绘制数据分布图
plt.scatter(X[:, 0], X[:, 1], c="red", marker="o", label="see")
plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.legend(loc=2)
plt.show()


def Model(n_clusters):
    estimator = KMeans(n_clusters=n_clusters)  # 构造聚类器
    return estimator


def train(estimator):
    estimator.fit(X)  # 聚类


# 初始化实例，并开启训练拟合
estimator = Model(3)
train(estimator)

label_pred = estimator.labels_  # 获取聚类标签
# 绘制k-means结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker="o", label="label0")
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker="*", label="label1")
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker="+", label="label2")
plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.legend(loc=2)
plt.show()
