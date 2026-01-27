import numpy as np
import pandas as pd
def pairwise_distances(df: pd.DataFrame) -> pd.DataFrame:
    data = df.values()
    diff = data[:,np.newaxis,:]-data[np.newaxis,:,:]
    D = np.sqrt(np.sum(diff**2,axis=1))
    return D

def knn_indices(X: np.array,k: int) -> np.array:
    D = pairwise_distances(X)
    np.fill_diagonal(D,np.inf)
    knn = np.argsort(D,axis=1)[:,:k]
    return knn

def neighbor_overlap(knn_orig: np.array, knn_proj: np.array) -> np.array:
    n,k = knn_orig.shape
    overlap = np.zeros(n)
    for i in range(n):
        set_orig = set(knn_orig[i])
        set_proj = set(knn_proj[i])
        overlap[i] = len(set_orig.intersection(set_proj)) / k
    return overlap

def average_neighbor_overlap(knn_orig: np.array, knn_proj:np.array) -> np.array:
    return neighbor_overlap(knn_orig,knn_proj).mean()