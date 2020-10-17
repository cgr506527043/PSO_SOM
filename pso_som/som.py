# -*- coding: utf-8 -*-
import numpy as np
import pylab as pl
from collections import Counter
import numpy.linalg as LA

# -----------------------获取中心样本索引------------------------------
def center_index(som_centers):
    # 获取中心样本索引
    medoids_idx = []
    for som_center in som_centers:
        distances = []
        for data in dataset_old:
            distance = LA.norm(np.array(data) - np.array(som_center), 2)
            distances.append(distance)
        medoids_idx.append(distances.index(min(distances)))
    return medoids_idx

# -----------------------获取som聚类中心-------------------------------
def get_center(category):
    sample_centers = []
    som_centers = []
    som_classes = set(category)
    medoids_idx = []
    # 获取聚类中心点（加权平方和）
    for i in som_classes:
        som_index = np.where(np.array(category) == i)
        # medoids_idx.append(random.choice(np.array(som_index).tolist()[0]))
        som_centers.append((np.sum(np.array(dataset_old)[som_index], axis=0) / len(np.array(dataset_old)[som_index])).tolist())

    # 获取中心样本索引
    medoids_idx = center_index(som_centers)

    # 获取中心样本坐标
    for i in medoids_idx:
        sample_centers.append(dataset_old[i].tolist())
    return medoids_idx, sample_centers

def loadData(filepath, has_id, class_position):
    '''
    :param filepath:
    :return:
    train_X = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i in f.readlines():
            train_X.append(list(map(float, i.split(",")[:-1])))
    return train_X
    '''
    with open(filepath) as f:
        # lines = (line.strip() for line in f if '?' not in line)
        lines = (line.strip() for line in f)
        dataset = np.loadtxt(lines, delimiter=',', dtype=np.str)
    # np.random.shuffle(dataset)
    # classes = list(dataset[:, -1])
    # dataset = np.asarray(np.delete(dataset, -1, axis=1), dtype=np.float).tolist()
    if class_position == 'first':
        classes = dataset[:, 0]
        dataset = np.delete(dataset, 0, axis = 1)
        dataset = np.asarray(dataset, dtype=np.float)
    else:
        classes = dataset[:, -1]
        dataset = np.delete(dataset, -1, axis = 1)
        dataset = np.asarray(dataset, dtype=np.float)
    if has_id:
        # Remove the first column (ID)
        dataset = np.delete(dataset, 0, axis = 1)
    return dataset, classes

def normal_X(X):
    """
    :param X:二维矩阵，N*D，N个D维的数据
    :return: 将X归一化的结果
    """
    N, D = X.shape
    for i in range(N):
        temp = np.sum(np.multiply(X[i], X[i]))
        X[i] /= np.sqrt(temp)
    return X


def normal_W(W):
    """
    :param W:二维矩阵，D*(n*m)，D个n*m维的数据
    :return: 将W归一化的结果
    """
    for i in range(W.shape[1]):
        temp = np.sum(np.multiply(W[:, i], W[:, i]))
        W[:, i] /= np.sqrt(temp)
    return W

# 神经网络
class SOM(object):
    def __init__(self, X, output, iteration, batch_size):
        """
        :param X: 形状是N*D,输入样本有N个,每个D维
        :param output: (n,m)一个元组，为输出层的形状是一个n*m的二维矩阵
        :param iteration:迭代次数
        :param batch_size:每次迭代时的样本数量
        初始化一个权值矩阵，形状为D*(n*m)，即有n*m权值向量，每个D维
        """
        self.X = X
        self.output = output
        self.iteration = iteration
        self.batch_size = batch_size
        self.W = np.random.rand(X.shape[1], output[0] * output[1])
        print("权值矩阵维度:",self.W.shape)
        # print("权值矩阵:", self.W)

    # 获取领域半径
    def GetN(self, t):
        """
        :param t:时间t, 这里用迭代次数来表示时间
        :return: 返回一个整数，表示拓扑距离，时间越大，拓扑邻域越小
        """
        a = min(self.output)        # a表示与输出结点有关的正常数
        return int(a - float(a) * t / self.iteration)

    # 求学习率
    def Geteta(self, t, n):
        """
        :param t: 时间t, 这里用迭代次数来表示时间
        :param n: 拓扑距离
        :return: 返回学习率，
        """
        return np.power(np.e, -n) / (t + 2)

    def getneighbor(self, index, N):
        """
        :param index:获胜神经元的下标
        :param N: 邻域半径
        :return ans: 返回一个集合列表，分别是不同邻域半径内需要更新的神经元坐标
        """
        a, b = self.output
        length = a * b      # 采用矩形邻域

        def distence(index1, index2):
            i1_a, i1_b = index1 // a, index1 % b  # //:向下取整; %:返回除法的余数;
            i2_a, i2_b = index2 // a, index2 % b
            return np.abs(i1_a - i2_a), np.abs(i1_b - i2_b)  # abs() 函数返回数字的绝对值。

        ans = [set() for i in range(N + 1)]
        for i in range(length):
            dist_a, dist_b = distence(i, index)
            if dist_a <= N and dist_b <= N:
                ans[max(dist_a, dist_b)].add(i)     # 切比雪夫距离
        return ans

    # 更新权值矩阵
    def updata_W(self, X, t, winner):
        N = self.GetN(t)  # 表示随时间变化的拓扑距离
        for x, i in enumerate(winner):
            # 邻域需要更新的神经元
            to_update = self.getneighbor(i[0], N)
            for j in range(N + 1):
                e = self.Geteta(t, j)  # 表示学习率
                for w in to_update[j]:
                    self.W[:, w] = np.add(self.W[:, w], e * (X[x, :] - self.W[:, w]))       # 更新连接权值


    def train(self):
        """
        train_Y:训练样本与形状为batch_size*(n*m)
        winner:一个一维向量，batch_size个获胜神经元的下标
        :return:返回值是调整后的W
        """
        count = 0
        while self.iteration > count:
            # 参数意思分别 是从a 中以概率P，随机选择3个, p没有指定的时候相当于是一致的分布
            # np.random.choice(a=5, size=3, replace=False, p=None)
            # 非一致的分布，会以多少的概率提出来
            # np.random.choice(a=5, size=3, replace=False, p=[0.2, 0.1, 0.3, 0.4, 0.0])
            # replacement 代表的意思是抽样之后还放不放回去，如果是False的话，那么出来的三个数都不一样，如果是
            # True的话， 有可能会出现重复的，因为前面的抽的放回去了。

            # train_X = self.X
            train_X = self.X[np.random.choice(self.X.shape[0], self.batch_size)]
            # print("训练向量:",train_X)
            normal_W(self.W)
            # print("归一化的权值矩阵:", self.W)
            normal_X(train_X)
            # print("归一化训练向量:", train_X)
            train_Y = train_X.dot(self.W)
            # print(train_Y.shape)
            winner = np.argmax(train_Y, axis=1).tolist()        # 获胜结点下标
            print("winner:", winner)

            # squeeze_winner = np.squeeze(winner).tolist()
            # print("winner length:", len(set(squeeze_winner)))
            #
            # som_medoids_idx, sample_centers = get_center(squeeze_winner)
            # print(som_medoids_idx)

            self.updata_W(train_X, count, winner)               # 更新获胜结点及周边结点权值
            count += 1
        return self.W

    # 输出获胜神经元结果
    def train_result(self):
        normal_X(self.X)
        train_Y = self.X.dot(self.W)
        winner = np.argmax(train_Y, axis=1).tolist()
        return winner


# 画图
def draw(C):
    colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm', ]
    for i in range(len(C)):
        coo_X = []  # x坐标列表
        coo_Y = []  # y坐标列表
        for j in range(len(C[i])):
            coo_X.append(C[i][j][0])
            coo_Y.append(C[i][j][1])
        pl.scatter(coo_X, coo_Y, marker='x', color=colValue[i % len(colValue)], label=i)

    pl.legend(loc='upper right')  # 图例位置
    pl.show()

dataset, classes = loadData(filepath="../Iris/iris.data", has_id=None, class_position='last')
dataset_old = dataset.copy()
classes_old = classes.copy()

dataset = np.mat(dataset)

som = SOM(dataset, (1, 3), 100, len(dataset))
layers = som.train()
res = som.train_result()
category =[j for i in res for j in i]
print("最终结果winner:",category)

som_medoids_idx, sample_centers = get_center(category)
print("聚类中心索引及坐标:", som_medoids_idx, sample_centers)

classify = {}
for i, win in enumerate(res):
    if not classify.get(win[0]):
        classify.setdefault(win[0], [i])
    else:
        classify[win[0]].append(i)
print(classify)

# for i, j in classify.items():
#     for k in j:
#         with open('./som_result/'+str(i)+'.txt',"a+", encoding="utf-8") as f:
#             f.write(str(classes[k])+'\n')


C = []  # 未归一化的数据分类结果
D = []  # 归一化的数据分类结果
for i in classify.values():
    C.append(np.array(dataset_old)[i].tolist())
    D.append(dataset[i].tolist())
draw(C)
draw(D)



# iris accuracy
diff = 0
for i,j in enumerate(category):
    if i < 50:
        if j != list(dict(Counter(category[:50]).most_common(1)).keys())[0]:
            diff += 1
    if i >=50 and i < 100:
        if j != list(dict(Counter(category[50:100]).most_common(1)).keys())[0]:
            diff += 1
    if i >= 100:
        if j != list(dict(Counter(category[145:163]).most_common(1)).keys())[0]:
            diff += 1
print(1-diff/150)
