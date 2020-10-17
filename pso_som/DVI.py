# -*- coding:utf-8 -*-
import numpy as np
import numpy.linalg as LA
def compute_DV_index(data, x, clusters, nc):
    cluster_list =[]
    b = []
    # ours
    for cluster in clusters.values():
        cluster_list.append(LA.norm(np.array(data)[cluster]-np.array(data)[np.array(list(clusters.values())).flatten().tolist()], ord=2, axis=1).tolist())
    cluster_dis = filter(lambda x: x!=0, np.array(cluster_list).flatten().tolist())
    # K-means
    # for cluster in clusters:
    #     cluster_list.append(LA.norm(np.array(cluster)-np.array(clusters), ord=2, axis=1).tolist())
    # cluster_dis = filter(lambda x: x!=0, np.array(cluster_list).flatten().tolist())

    min_cluster_dis = min(cluster_dis)

    for i in x.values():
        sum = np.array(data)[i].sum(axis=0)
        avg_sum = sum / len(i)
        b.append(LA.norm(np.array(data)[i] - avg_sum, ord=2, axis=1).sum()/len(np.array(data)[i]))
    return min_cluster_dis/max(b)
