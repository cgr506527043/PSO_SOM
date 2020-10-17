# -*- coding:utf-8 -*-
import numpy as np
import random
import som
import math as m
import numpy.linalg as LA
from numpy import *
from collections import Counter

def center_index(data, centers):
    medoids_idx = []
    for i in centers:
        distances = []
        for j in data:
            distance = LA.norm(np.array(j) - np.array(centers[i]), 2)
            distances.append(distance)
        medoids_idx.append(distances.index(min(distances)))
    return medoids_idx

def center(data, clusterRes, k):
    clunew = {}
    for i in range(k):
        # 计算每个组的新质心
        if clusterRes[i]:
            idx = np.array(clusterRes[i])  # numpy.where(condition) 输出满足条件的元素的坐标
            sum = np.array(data)[idx].sum(axis=0)
            avg_sum = sum / len(np.array(data)[idx])
            clunew[i]=avg_sum
    # 获取中心样本索引
    medoids_idx = center_index(data, clunew)
    centroids={}
    # 获取中心样本坐标(字典形式)
    for i in clusterRes:
        for j in medoids_idx:
            if j in clusterRes[i]:
                centroids[i]=j
    return medoids_idx, centroids

# 计算样本与质点的距离
def cal_dis(data, clu):
    """
    计算质点与数据点的距离
    :param data: 样本点
    :param clu:  质点集合
    :param k: 类别个数
    :return: 质心与样本点距离矩阵
    """
    dis = []
    for i in range(len(data)):
        dis.append([])
        for j in clu:
            dis[i].append(m.sqrt(sum(pow(data[i, :] - data[clu[j]], 2))))
    return np.asarray(dis)

# 划分聚类簇
def divide(data, dis):
    """
    对数据点分组
    :param data: 样本集合
    :param dis: 质心与所有样本的距离
    :param k: 类别个数
    :return: 分割后样本
    """
    clusterRes = {}
    for i in range(len(data)):
        seq = np.argsort(dis[i])
        if seq[0] not in list(clusterRes.keys()):
            clusterRes.setdefault(seq[0], [i])
        else:
            clusterRes[seq[0]].append(i)
        # clusterRes[str(seq[0])].append(i)
    return clusterRes

def classfy(data, clu):
    """
    迭代收敛更新质心
    :param data: 样本集合
    :param clu: 质心集合
    :param k: 类别个数
    :return: 误差， 新质心
    """
    # 样本与旧质心的距离
    clulist = cal_dis(data, clu)
    # 样本分组
    clusterRes = divide(data, clulist)
    # 更新质心
    medoids_idx, clunew = center(data, clusterRes, len(clusterRes))
    return medoids_idx, clunew, clusterRes


# ----------------------PSO参数设置---------------------------------
class PSO():
    def __init__(self, data, classify, centroids, pN, dim, max_iter):
        self.dataset = data
        self.classify = classify
        self.centroids = centroids
        self.w = 0.8       # 惯性权重
        self.w_max = 1    # 最大权重
        self.w_min = 0.4  # 最小权重

        self.c1 = 2        # 个体学习因子，一般取值为2
        self.c2 = 2         # 全局学习因子

        self.r1 = 0.8
        self.r2 = 0.2

        self.pN = pN        # 粒子群数量
        self.dim = dim      # 搜索维度
        self.max_iter = max_iter        # 迭代次数

        self.X = np.zeros((self.pN, self.dim))      # 所有粒子的位置和速度
        self.V = np.zeros((self.pN, self.dim))

        self.pbest = np.zeros((self.pN, self.dim))      # 个体经历的最佳位置和全局最佳位置
        self.gbest = np.zeros((1, self.dim))

        self.p_fit = np.zeros(self.pN)      # 每个个体的历史最佳适应值
        self.fit = 1e10     # 全局最佳适应值

    # ---------------------初始化种群----------------------------------
    def init_Population(self):
        for i in range(self.pN):
            for j in range(self.dim):
                self.X[i][j] = self.dataset[i][j]
                self.V[i][j] = random.uniform(0, 1)

            self.pbest[i] = self.X[i]
            tmp = self.function(self.X[i], i)
            self.p_fit[i] = tmp
            if (tmp < self.fit):
                self.fit = tmp
                self.gbest = self.X[i]

    # ----------------------权重线性减少--------------------------------
    def reduce_liner_weight(self, iter):
        self.w = self.w_max - iter * (self.w_max - self.w_min) / self.max_iter
        return self.w
    # ----------------------权重非线性减少------------------------------
    def reduce_noliner_weight(self, iter):
        self.w = self.w_max - (self.w_max - self.w_min)*m.log(1+iter/self.max_iter)
        return self.w

# ---------------------适应度函数-----------------------------
    def function(self, x, i):
        # 未知聚类类标，自己打标
        distances_sum = 0
        for m, n in self.classify.items():
            if i in n:
                for centroid_key in self.centroids:
                    if centroid_key != m:
                        distances_sum += LA.norm(np.array(self.dataset[self.centroids[centroid_key]])-np.array(x),2)
        return 1/distances_sum

# ----------------------更新粒子位置----------------------------------
    def iterator(self):
        fitness = []
        f = open("pso_kmeans.txt", 'w', encoding='utf-8')
        for t in range(self.max_iter):
            # self.w = self.reduce_liner_weight(t)
            # self.w = self.reduce_noliner_weight(t)
            for i in range(self.pN):  # 更新gbest\pbest
                temp = self.function(self.X[i], i)
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

            # 重新计算聚类中心和聚类划分
            # clusterRes: classify
            # clunew: centroids

            idx, self.centroids, self.classify = classfy(self.X, self.centroids)
            print(self.classify)
            f.write('第%d轮迭代-------------\n' % (t))
            f.write(str(self.centroids)+'\n')
            f.write(str(self.classify) + '\n')

            acc = 0
            for i in self.classify:
                acc += Counter(np.array(classes)[self.classify[i]]).most_common(1)[0][1]
            print(acc/150)

        return fitness, self.gbest

def nearestCentroidClassifier(pso_centroids):
    error = 0
    pso_centers = list(pso_centroids.values())
    for i, j in enumerate(som.dataset_old):
        dis = LA.norm(np.asarray(j)-np.asarray(pso_centers), 2, axis=1).tolist()
        if som.classes[i] != list(pso_centroids.keys())[dis.index(min(dis))]:
            error += 1
    return error

classify, classes = som.classify, som.classes

som_medoids_idx, som_centroids = center(som.dataset_old, classify, len(set(classify)))
print("som最终聚类中心:", list(som_centroids.keys()), som_centroids)

my_pso = PSO(data=som.dataset_old, classify=classify, centroids=som_centroids, pN=len(som.dataset_old), dim=np.array(som.dataset_old).shape[1], max_iter=100)
my_pso.init_Population()
print("初始化成功!")

fitness, last_gbest = my_pso.iterator()
print(fitness)

'''
import matplotlib.pyplot as plt

plt.plot(list(range(len(fitness))), fitness, color='green', label='Ours')
plt.legend() # 显示图例
plt.show()
'''
