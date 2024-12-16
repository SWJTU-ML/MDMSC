import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial.distance import pdist, squareform
from src.evaluation import compute_score
from src.MDMSC import MDMSC
import scipy.io as scio
import time
import os
from scipy.io import arff
import sys
np.seterr(invalid='ignore')

def plot(data, label):
    colors = ['tomato', 'deepskyblue', 'hotpink', 'yellowgreen', 'mediumorchid', 'gold', 'orchid', 'limegreen',
              'lightcoral']
    plt.figure(figsize=(12, 8))
    for point in range(len(data)):
        cluster_label = label[point]  
        cluster_color = colors[cluster_label % len(colors)]  
        plt.scatter(data[point, 0], data[point, 1], c=cluster_color)  
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Ours", fontsize=30)
    plt.show()


def syn():
    data_ = np.loadtxt('dataset/synthesis/Jain.txt')
    data = data_[:, :-1]
    y = data_[:, -1]

    # data1 = scio.loadmat('dataset/synthesis/diamond9.mat')
    # data_keys = list(data1.keys())  
    # print(data_keys)
    # data = data1[data_keys[-2]]
    # y = data1[data_keys[-1]]

    data = data.astype(np.float32)
    n, d = data.shape
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    y = y.reshape(y.shape[0])
    n_clusters = len(np.unique(y))
    print(n, d, n_clusters)
    dist = squareform(pdist(data))
    for k in range(10,11):
        lambda_ratio = 1.5
        belta = 8
        label = MDMSC(data, y, n_clusters, k, dist,lambda_ratio,belta)
        ARI, NMI, ACC = compute_score(label, y)  
        print('k: {d:.2f}\nARI: {a:.2f}\nNMI: {b:.2f}\nACC: {c:.2f}'.format(d=k, a=ARI, b=NMI, c=ACC))



def real():

    # data1 = scio.loadmat('dataset/real_all/mfea-fac.mat')
    # data_keys = list(data1.keys())  
    # data = data1[data_keys[-2]]
    # y = data1[data_keys[-1]]

    data_ = np.loadtxt('dataset/arff_txt/optical_test.txt')
    data = data_[:, :-1]
    y = data_[:, -1]

    data = data.astype(np.float32)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    n, d = data.shape
    print(n, d)

    y = np.ravel(y)
    n_clusters = len(np.unique(y))
    dist = squareform(pdist(data))
    lambda_ratio=1.5
    belta=16
    for k in range(3,5):
        label= MDMSC(data, y, n_clusters, k, dist,lambda_ratio,belta)
        ARI, NMI, ACC = compute_score(label, y)  
        print('k: {d:.2f}\tARI: {a:.2f}\tNMI: {b:.2f}\tACC: {c:.2f}'.format(d=k, a=ARI, b=NMI, c=ACC))

real()
# syn()
