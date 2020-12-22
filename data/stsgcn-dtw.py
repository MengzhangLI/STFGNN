import argparse
import numpy as np
import pandas as pd
#from fastdtw import compute_dtw
import time

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="PEMS04", help="Dataset path.")
parser.add_argument("--order", type=int, default=1, help="DTW order.")
parser.add_argument("--lag", type=int, default=12, help="Fast DTW search lag.")
parser.add_argument("--period", type=int, default=288, help="Time series perios.")

args = parser.parse_args()

def normalize(a):
    mu=np.mean(a,axis=1,keepdims=True)
    std=np.std(a,axis=1,keepdims=True)
    return (a-mu)/std

def compute_dtw(a,b,order=1,Ts=12,normal=True):
    if normal:
        a=normalize(a)
        b=normalize(b)
    T0=a.shape[1]
    d=np.reshape(a,[-1,1,T0])-np.reshape(b,[-1,T0,1])
    d=np.linalg.norm(d,axis=0,ord=order)
    D=np.zeros([T0,T0])
    for i in range(T0):
        for j in range(max(0,i-Ts),min(T0,i+Ts+1)):
            if (i==0) and (j==0):
                D[i,j]=d[i,j]**order
                continue
            if (i==0):
                D[i,j]=d[i,j]**order+D[i,j-1]
                continue
            if (j==0):
                D[i,j]=d[i,j]**order+D[i-1,j]
                continue
            if (j==i-Ts):
                D[i,j]=d[i,j]**order+min(D[i-1,j-1],D[i-1,j])
                continue
            if (j==i+Ts):
                D[i,j]=d[i,j]**order+min(D[i-1,j-1],D[i,j-1])
                continue
            D[i,j]=d[i,j]**order+min(D[i-1,j-1],D[i-1,j],D[i,j-1])
    return D[-1,-1]**(1.0/order)


df = np.load(args.dataset+'/'+args.dataset+".npz")['data']
num_samples,ndim,_ = df.shape
num_train = int(num_samples * 0.6)
num_dtw=int(num_train/args.period)*args.period
data=df[:num_dtw,:,:1].reshape([-1,args.period,ndim])

d=np.zeros([ndim,ndim])

for i in range(ndim):
    t1=time.time()
    for j in range(i+1,ndim):
        d[i,j]=compute_dtw(data[:,:,i],data[:,:,j],order=args.order,Ts=args.lag)
    t2=time.time()
    print('Line',i,'finished in',t2-t1,'seconds.')

dtw=d+d.T

np.save("stsgcn/"+args.dataset+"-dtw-"+str(args.period)+'-'+str(args.order)+"-.npy",dtw)

