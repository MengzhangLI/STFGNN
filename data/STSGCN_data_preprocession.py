import numpy as np 
import pandas as pd 


df = pd.read_csv("./PeMSD7_V_228.csv", header=None)
df = np.array(df)
df_new = df.reshape([df.shape[0], df.shape[1], 1])
filename = './PeMSD7M.npz'
np.savez(filename, data = df_new)


adj = pd.read_csv("./PeMSD7_W_228.csv", header=None)

adj_arr = []
for i in range(adj.shape[0]):
    for j in range(adj.shape[1]):
        if adj[i][j] != 0 and adj[i][j] <= 2000:
            adj_arr.append([i, j, adj[i][j]])

adj_new = pd.DataFrame(adj_arr, columns=['from','to', 'cost'])

adj_new.to_csv('./PeMSD7M.csv', index=False)
