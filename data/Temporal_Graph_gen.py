#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import time

def gen_data(data, ntr, N):
    '''
    if flag:
        data=pd.read_csv(fname)
    else:
        data=pd.read_csv(fname,header=None)
    '''
    #data=data.as_matrix()
    data=np.reshape(data,[-1,288,N])
    return data[0:ntr]

def normalize(a):
    mu=np.mean(a,axis=1,keepdims=True)
    std=np.std(a,axis=1,keepdims=True)
    return (a-mu)/std

def compute_dtw(a,b,o=1,T=12):
    a=normalize(a)
    b=normalize(b)
    d=np.reshape(a,[-1,1,T0])-np.reshape(b,[-1,T0,1])
    d=np.linalg.norm(d,axis=0,ord=o)
    D=np.zeros([T0,T0])
    for i in range(T0):
        for j in range(max(0,i-T),min(T0,i+T+1)):
            if (i==0) and (j==0):
                D[i,j]=d[i,j]**o
                continue
            if (i==0):
                D[i,j]=d[i,j]**o+D[i,j-1]
                continue
            if (j==0):
                D[i,j]=d[i,j]**o+D[i-1,j]
                continue
            if (j==i-T):
                D[i,j]=d[i,j]**o+min(D[i-1,j-1],D[i-1,j])
                continue
            if (j==i+T):
                D[i,j]=d[i,j]**o+min(D[i-1,j-1],D[i,j-1])
                continue
            D[i,j]=d[i,j]**o+min(D[i-1,j-1],D[i-1,j],D[i,j-1])
    return D[-1,-1]**(1.0/o)

data = np.load('./PEMS08/PEMS08.npz')['data'][:,:,0]
total_day = data.shape[0] / 288
tr_day = int(total_day * 0.6)
n_route = data.shape[1]

xtr = gen_data(data, tr_day, n_route)
print(np.shape(xtr))
T0 = 288
#o = 2
T = 12
N = n_route #325
d = np.zeros([N, N])
for i in range(N):
    t1=time.time()
    for j in range(i+1,N):
        d[i,j]=compute_dtw(xtr[:,:,i],xtr[:,:,j])
    t2=time.time()
    print(t2-t1)
    print("=======================")
np.save("./fast_test_PeMS08.npy", d)
print("The calculation of time series is done!")

adj = np.load("./fast_test_PeMS08.npy")
adj = adj+ adj.T
n = adj.shape[0]

w_adj = np.zeros([n,n])
adj_percent = 0.01
top = int(n * adj_percent)
for i in range(adj.shape[0]):
    a = adj[i,:].argsort()[0:top]
    for j in range(top):
        w_adj[i, a[j]] = 1

for i in range(n):
    for j in range(n):
        if (w_adj[i][j] != w_adj[j][i] and w_adj[i][j] ==0):
            w_adj[i][j] = 1
        if( i==j):
            w_adj[i][j] = 1
print("Total route number: ", n)
print(len(w_adj.nonzero()[0])/(n*n))
ww = pd.DataFrame(w_adj)
ww.to_csv("./adj_PEMS08_001.csv", index = False, header=None)
print("The weighted matrix of temporal graph is generated!")