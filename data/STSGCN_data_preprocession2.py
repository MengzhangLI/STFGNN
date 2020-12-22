import numpy as np 
import pandas as pd 
import pickle

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def df_pad(df):
    df.replace(0, np.nan, inplace=True)
    temp1 = df.bfill(axis=0)
    temp2 = temp1.ffill(axis=0)

    return temp2

def df_fill(df):
    df_mean = df[df !=0 ].mean()
    df.replace(0, df_mean, inplace=True)

    # Set 0 on Test Set for fair comparison.
    test_ix = int(df.shape[0] * 0.2)
    df.iloc[-test_ix:, :].replace(df_mean, 0, inplace = True)

    return df

#df = pd.read_hdf("./metr-la.h5")
df = pd.read_hdf("./pems-bay.h5")

#filename = './METR-LA/metr-la_pad.npz'
#filename = './PEMS-BAY/pems-bay_pad.npz'
#filename = './METR-LA/metr-la_fill.npz'
filename = './PEMS-BAY/pems-bay_fill.npz'

#pkl_filename = './sensor_graph/adj_mx.pkl'
pkl_filename = './sensor_graph/adj_mx_bay.pkl'


#df = df_pad(df)
df = df_fill(df)

df = np.array(df)
df_new = df.reshape([df.shape[0], df.shape[1], 1])

np.savez(filename, data = df_new)

#adj = pd.read_csv("./PeMSD7_W_228.csv", header=None)
sensor_ids, sensor_id_to_ind, adj = load_pickle(pkl_filename)
adj_arr = []
print("Spatial Adj Sparsity: ", np.sum((adj!=0).astype(int))/ (adj.shape[0] * adj.shape[1]))
for i in range(adj.shape[0]):
    for j in range(adj.shape[1]):
        #if adj[i][j] != 0 and adj[i][j] <= 2000:
        if adj[i][j] != 0:
            adj_arr.append([i, j, adj[i][j]])

adj_new = pd.DataFrame(adj_arr, columns=['from','to', 'cost'])
#adj_new.to_csv('./METR-LA/metr-la.csv', index=False)
#adj_new.to_csv('./PEMS-BAY/pems-bay.csv', index=False)