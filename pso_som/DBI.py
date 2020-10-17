# -*- coding:utf-8 -*-
import math
import numpy.linalg as LA
import numpy as np

# nc is number of clusters
# to be implemented without the use of any libraries (from the scratch)

def vectorDistance(v1, v2):
    """
    this function calculates de euclidean distance between two
    vectors.
    """
    return LA.norm(np.array(v1)-np.array(v2), ord=2)
    # sum = 0
    # for i in range(len(v1)):
    #     sum += (v1[i] - v2[i]) ** 2
    # return sum ** 0.5


def compute_Si(data, x, clusters):
    # ours
    for i in x.values():
        if data.tolist().index(clusters.tolist()) in i:
            return LA.norm(np.array(data)[i]-clusters, ord=2, axis=1).sum()/len(np.array(data)[i])

    # K-means  pso
    # for i in x.values():
    #     if data.index(clusters.tolist()) in i:
    #         return LA.norm(np.array(data)[i]-clusters, ord=2, axis=1).sum()/len(np.array(data)[i])

    # norm_c = nc
    # s = 0
    # for t in x[i]:
    #     s += vectorDistance(t, clusters)
    # return s / norm_c


def compute_Rij(i, j, data, x, clusters, nc):

    # K-means  pso
    # Mij = vectorDistance(clusters[i], clusters[j])
    # Rij = (compute_Si(data, x, clusters[i]) + compute_Si(data, x, clusters[j])) / Mij


    # Ours
    Mij = vectorDistance(data[clusters[i]], data[clusters[j]])
    Rij = (compute_Si(data, x, data[clusters[i]]) + compute_Si(data, x, data[clusters[j]])) / Mij

    return Rij


def compute_Di(i, data, x, clusters, nc):
    list_r = []
    for j in range(nc):
        if i != j:
            temp = compute_Rij(i, j, data, x, clusters, nc)
            list_r.append(temp)
    return max(list_r)


def compute_DB_index(data, x, clusters, nc):
    sigma_R = 0.0
    for i in range(nc):
        sigma_R = sigma_R + compute_Di(i, data, x, clusters, nc)
    DB_index = float(sigma_R) / float(nc)
    return DB_index
