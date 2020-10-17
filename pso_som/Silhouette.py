# -*- coding:utf-8 -*-
import numpy as np
import numpy.linalg as LA

def compute_Silhouette(data, x):
    sc = []
    for i in range(len(data)):
        for j, k in x.items():
            if i in k:
                inner_dis = LA.norm(np.array(data)[i]-np.array(data)[k], ord=2, axis=1).sum()
                keys = list(x.keys())
                keys.remove(j)
                outter = []
                for m in keys:
                    outter.append(LA.norm(np.array(data)[x[m]]-np.array(data)[i], ord=2, axis=1).sum().tolist())
                outter_dis = min(outter)
                if outter_dis > inner_dis:
                    sc.append(1-inner_dis/outter_dis)
                else:
                    sc.append(outter_dis/inner_dis-1)
    return sum(sc)/len(data)




