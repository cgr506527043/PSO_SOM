# -*- coding:utf-8 -*-
import numpy as np
import numpy.linalg as LA
import math as m
from sklearn.cluster import KMeans
import pandas as pd
import DBI
import DVI
import Silhouette

def loadData(filepath, has_id, class_position):
    with open(filepath) as f:
        lines = (line.strip() for line in f)
        dataset = np.loadtxt(lines, delimiter=',', dtype=np.str, comments="#")
    if has_id:
        # Remove the first column (ID)
        dataset = np.delete(dataset, 0, axis = 1)
    if class_position == 'first':
        classes = dataset[:, 0]
        dataset = np.delete(dataset, 0, axis = 1)
        dataset = np.asarray(dataset, dtype=np.float)
    else:
        classes = dataset[:, -1]
        dataset = np.delete(dataset, -1, axis = 1)
        dataset = np.asarray(dataset, dtype=np.float)
    return dataset, classes

def center_index(data, centers):
    medoids_idx = []
    for i in list(centers.values()):
        distances = []
        for j in data:
            distance = LA.norm(np.array(j) - np.array(i), 2)
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
    print(medoids_idx)

    centroids={}
    # 获取中心样本坐标(字典形式)
    for i in clusterRes:
        for j in medoids_idx:
            if j in clusterRes[i]:
                centroids[i]=dataset[j]
    return centroids

def cal_dis(data, clu):
    """
    计算质点与数据点的距离
    :param data: 样本点
    :param clu:  粒子集合
    :param k: 类别个数
    :return: 质心与样本点距离矩阵
    """
    dis = []
    for i in data:
        dis.append(LA.norm(i - clu.reshape(k, d), 2, axis=1).tolist())
    return dis
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
    return clusterRes

def classfy(data, clu):
    """
    迭代收敛更新质心
    :param data: 样本集合
    :param clu: 单个粒子
    :param k: 类别个数
    :return: 误差， 新质心
    """
    # 样本与旧质心的距离
    clulist = cal_dis(data, clu)
    # 样本分组
    clusterRes = divide(data, clulist)
    # 更新质心
    clunew = center(data, clusterRes, len(clusterRes))

    return np.concatenate(list(clunew.values())), clusterRes

# ----------------------PSO参数设置---------------------------------
class PSO():
    def __init__(self, data, init_classify, centroids, pN, dim, max_iter):
        self.dataset = data
        self.init_classify = init_classify
        self.classify = []
        self.DBI = []
        self.DVI = []
        self.Silhouette = []
        self.centroids = centroids


        self.w = 0.8       # 惯性权重
        self.w_max = 1   # 最大权重
        self.w_min = 0.4   # 最小权重

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
            self.X[i] = np.concatenate(list(self.centroids.values()))
            self.V[i] = np.random.uniform(0, 1, self.dim)
            self.classify.append(self.init_classify)
            self.pbest[i] = self.X[i]
            tmp = self.function(self.X[i], self.classify[i])
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
    def function(self, x, classify_one):
        distances_sum = 0
        for centroid in x.reshape(k,d):
            for j in list(classify_one.values()):
                if dataset.index(centroid.tolist()) in j:
                    distances_sum += LA.norm(np.array(centroid) - np.array(dataset)[j], ord=2, axis=1).sum()
        return distances_sum


# ----------------------更新粒子位置----------------------------------
    def iterator(self):
        fitness = []
        for t in range(self.max_iter):
            self.w = self.reduce_liner_weight(t)
            # self.w = self.reduce_noliner_weight(t)
            for i in range(self.pN):                # 更新gbest\pbest
                temp = self.function(self.X[i], self.classify[i])
                if (temp < self.p_fit[i]):                  # 更新个体最优
                    self.p_fit[i] = temp
                    self.pbest[i] = self.X[i]
                    if (self.p_fit[i] < self.fit):              # 更新全局最优
                        self.gbest = self.X[i]
                        self.fit = self.p_fit[i]
            fitness.append(self.fit)
            print(self.fit)
            pso_medoids_idx = []
            for j in self.gbest.reshape(k, d):
                pso_medoids_idx.append(dataset.index(j.tolist()))

            pso_centroids = {}
            for idx in pso_medoids_idx:
                pso_centroids[classes[idx]] = dataset[idx]
            error = nearestCentroidClassifier(pso_centroids)
            print("第{i}轮迭代的准确率:{accuracy}".format(i=t, accuracy=1 - error / len(self.dataset)))

            for i in range(self.pN):
                self.V[i] = self.w * self.V[i] + self.c1 * self.r1 * (self.pbest[i] - self.X[i]) + \
                            self.c2 * self.r2 * (self.gbest - self.X[i])
                self.X[i] = self.X[i] + self.V[i]
                self.X[i], self.classify[i] =classfy(self.dataset, self.X[i])
                print(self.classify[i])
                self.DBI.append(DBI.compute_DB_index(self.dataset, self.classify[i], self.X[i].reshape(k,d), k))
                self.DVI.append(DVI.compute_DV_index(self.dataset, self.classify[i], self.X[i].reshape(k, d), k))
                self.Silhouette.append(Silhouette.compute_Silhouette(self.dataset, self.classify[i]))
            # 重新计算聚类中心和聚类划分
            # clusterRes: classify
            # clunew: centroids
        print("DBI:",min(self.DBI))
        print("DVI:",max(self.DVI))
        print("Silhouette:",max(self.Silhouette))
        return fitness, self.gbest

def nearestCentroidClassifier(pso_centroids):
    error = 0
    pso_centers = list(pso_centroids.values())
    for i, j in enumerate(dataset):
        dis = LA.norm(np.asarray(j)-np.asarray(pso_centers), 2, axis=1).tolist()
        if classes[i] != list(pso_centroids.keys())[dis.index(min(dis))]:
            error += 1
    return error

k = 11                         # number of clusters
d = 768                         # dimension for each data row
N = k*d                                             # dimension of each particle
nparticles = 5                                     # number of particles
maxiter = 10                                       # number of iteractions

dataset = pd.read_csv("Word.txt", sep=",", header=None).ix[:, 0:768].values.tolist()
print(dataset)
classes = pd.read_csv("Word.txt", sep=",", header=None).ix[:,768].values.tolist()
print(classes)

classify = {}
for i, win in enumerate(classes):
    if not classify.get(win):
        classify.setdefault(win, [i])
    else:
        classify[win].append(i)
centroids = {}
for j in classify.values():
    m = np.random.choice(j)
    centroids[m] = dataset[m]
print(centroids)

my_pso = PSO(dataset, classify, centroids, nparticles, N, maxiter)
my_pso.init_Population()
print("初始化成功!")
fitness, last_gbest= my_pso.iterator()
print("pso最佳位置:",last_gbest)
# with open("pso-km.txt", "w", encoding="utf-8") as f:
#     for i in fitness:
#         f.write(str(i)+"\n")



# import matplotlib.pyplot as plt
#
# plt.plot(list(range(len(fitness))), fitness, color='blue', label='pso-km')
# plt.legend() # 显示图例
# plt.show()


'''
k = len(som.classify)                                  # number of clusters
d = len(som.dataset_old[0])                         # dimension for each data row
N = k*d                                             # dimension of each particle
nparticles = 10                                     # number of particles
maxiter = 100                                       # number of iteractions

som_medoids_idx, som_centroids = som.som_medoids_idx, som.map_centers       # som聚类中心索引和坐标

my_pso = PSO(som.dataset_old, som.classify, dict(zip(som_medoids_idx, som_centroids)), nparticles, N, maxiter)
my_pso.init_Population()
print("初始化成功!")
fitness, last_gbest = my_pso.iterator()
print("pso最佳位置:",last_gbest)
'''



