# -*- coding:utf-8 -*-
import numpy as np
import random
import som
import numpy.linalg as LA
from numpy import *
import pylab as pl

# -----------------------获取中心样本索引------------------------------
def center_index(som_centers):
    # 获取中心样本索引
    medoids_idx = []
    for som_center in som_centers:
        distances = []
        for data in som.dataset_old:
            distance = LA.norm(np.array(data) - np.array(som_center), 2)
            distances.append(distance)
        medoids_idx.append(distances.index(min(distances)))
    return medoids_idx

# -----------------------获取som聚类中心-------------------------------
def get_center():
    sample_centers = []
    som_centers = []
    som_classes = set(som.category)
    medoids_idx = []
    # 获取聚类中心点（加权平方和）
    for i in som_classes:
        som_index = np.where(np.array(som.category) == i)
        # medoids_idx.append(random.choice(np.array(som_index).tolist()[0]))
        som_centers.append((np.sum(np.array(som.dataset_old)[som_index], axis=0) / len(np.array(som.dataset_old)[som_index])).tolist())

    # 获取中心样本索引
    medoids_idx = center_index(som_centers)

    # 获取中心样本坐标
    for i in medoids_idx:
        sample_centers.append(som.dataset_old[i])
    return medoids_idx, sample_centers

# 欧式距离
def euclidianDistance(a, b):
    diff_sqrt = [(x - y)**2 for x, y in zip(a, b)]
    return np.sqrt(np.sum(diff_sqrt))

# ----------------------PSO参数设置---------------------------------
class PSO():
    def __init__(self, pN, dim1, dim2, max_iter):
        self.w = 0        # 惯性权重
        self.w_max = 1
        self.w_min = 0.4
        self.c1 = 2        # 个体学习因子，一般取值为2
        self.c2 = 2         # 全局学习因子
        self.r1 = 0.8
        self.r2 = 0.2
        self.pN = pN        # 粒子群数量
        self.dim1 = dim1      # 搜索维度
        self.dim2 = dim2
        self.max_iter = max_iter        # 迭代次数
        self.X = np.zeros((self.pN, self.dim1, self.dim2))      # 所有粒子的位置和速度
        self.V = np.zeros((self.pN, self.dim1, self.dim2))
        self.pbest = np.zeros((self.pN, self.dim1, self.dim2))      # 个体经历的最佳位置和全局最佳位置
        self.gbest = np.zeros((1, self.dim1, self.dim2))
        self.p_fit = np.zeros(self.pN)      # 每个个体的历史最佳适应值
        self.fit = 1e10     # 全局最佳适应值

    # ---------------------初始化种群----------------------------------
    def init_Population(self):
        for i in range(self.pN):        # 50
            for j in range(self.dim1):      # 3
                for k in range(self.dim2):      # 4
                    self.X[i][j][k] = sample_centers[j][k]
                    self.V[i][j][k] = random.uniform(0, 1)

            self.pbest[i] = self.X[i]
            tmp = self.function(self.X[i])
            self.p_fit[i] = tmp
            if (tmp < self.fit):
                 self.fit = tmp
                 self.gbest = self.X[i]

    # ----------------------权重线性减少--------------------------------
    def reduce_weight(self, iter):
        self.w = self.w_max - iter * (self.w_max-self.w_min) / self.max_iter
        return self.w


    # ---------------------适应度函数-----------------------------
    def function(self, x):
        cost = {}
        distances_sum = 0
        count = 0
        for k in x:
            distance = LA.norm(np.array(k) - np.array(som.dataset_old), axis=1).tolist()
            index = distance.index(min(distance))
            for i, j in enumerate(som.dataset_old):
                if str(som.classes_old[i]) == str(som.classes[index]):
                    distances_sum += euclidianDistance(j, k)
                    count += 1
            cost[str(som.classes[index])] = distances_sum
        cost_sum = sum(list(cost.values()))
        return cost_sum


# ----------------------更新粒子位置----------------------------------
    def iterator(self):
        fitness = []
        for t in range(self.max_iter):
            self.w = self.reduce_weight(t)
            for i in range(self.pN):  # 更新gbest\pbest
                temp = self.function(self.X[i])
                if (temp < self.p_fit[i]):  # 更新个体最优
                    self.p_fit[i] = temp
                    self.pbest[i] = self.X[i]
                    if (self.p_fit[i] < self.fit):  # 更新全局最优
                        self.gbest = self.X[i]
                        self.fit = self.p_fit[i]
            for i in range(self.pN):
                self.V[i] = self.w * self.V[i] + self.c1 * self.r1 * (self.pbest[i] - self.X[i]) + \
                            self.c2 * self.r2 * (self.gbest - self.X[i])
                self.X[i] = self.X[i] + self.V[i]
            fitness.append(self.fit)
            # print("群体最优值:", self.fit)  # 输出最优值
            print(self.fit)
            pso_medoids_idx = center_index(self.gbest)
            pso_centroids = {}
            for idx in pso_medoids_idx:
                pso_centroids[som.classes_old[idx]] = som.dataset_old[idx]
            error = nearestCentroidClassifier(pso_centroids)
            print("第{i}轮迭代的准确率:{accuracy}".format(i=t, accuracy=1 - error / 150))
        return fitness, self.gbest

def nearestCentroidClassifier(pso_centroids):
    error = 0
    pso_centers = list(pso_centroids.values())
    for i, j in enumerate(som.dataset_old):
        dis = LA.norm(np.asarray(j)-np.asarray(pso_centers), 2, axis=1).tolist()
        if som.classes[i] != list(pso_centroids.keys())[dis.index(min(dis))]:
            error += 1
    return error



som_medoids_idx, sample_centers = get_center()
print("聚类中心索引及坐标:", som_medoids_idx, sample_centers)

centroids = {}
for i in sample_centers:
    centroids[som.classes_old[som.dataset_old.index(i)]] = i
print("som聚类中心:", centroids)

my_pso = PSO(pN=3, dim1=len(sample_centers), dim2=len(sample_centers[0]), max_iter=100)
my_pso.init_Population()
fitness, last_centers = my_pso.iterator()
print("pso聚类中心:",last_centers)

pso_medoids_idx = center_index(last_centers)
pso_centroids = {}
for idx in pso_medoids_idx:
    pso_centroids[som.classes_old[idx]] = som.dataset_old[idx]
print(pso_centroids)
error = nearestCentroidClassifier(pso_centroids)
print("最终准确率:", 1-error/150)


