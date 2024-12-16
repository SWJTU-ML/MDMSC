import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from sklearn.cluster import KMeans
from scipy.spatial.distance import squareform, pdist


def ger_rho(data, k):
    n = data.shape[0]
    nbr = NearestNeighbors(n_neighbors=k + 1).fit(data)
    distances, indices = nbr.kneighbors(data)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    distances = distances / np.max(distances)
    rho = np.sum(np.exp(-distances ** 2), axis=1)
    return rho, distances, indices


def find_core(rho, nbrs):
    core = []
    n = rho.shape[0]
    for i in range(n):
        flag = 1
        for j in nbrs[i]:
            if rho[j] > rho[i]:
                flag = 0
                break
        if flag == 1:
            core.append(i)
    return core


def find_leader(data, rho, nbrs, core, dist):
    n = data.shape[0]
    leader = np.full((n, 1), -1)
    delta = np.zeros((n, 1))
    rho_descend = np.argsort(-rho)
    for i, index in enumerate(rho_descend):
        if index in core:
            continue
        ascend_rho = rho_descend[:i]
        delta[index] = np.min(dist[index, ascend_rho])
        leader[index] = ascend_rho[np.argmin(dist[index, ascend_rho])]
    return delta, leader


def get_sub_cluster(data, core, leader):
    n = data.shape[0]
    sub_clusters = [[] for i in range(len(core))]
    clustered = np.full((n, 1), -1)
    for i, center in enumerate(core):
        sub_clusters[i].append(center)
        clustered[center] = i
        j = 0
        while j < len(sub_clusters[i]):
            p = sub_clusters[i][j]
            points = np.where(leader == p)[0]
            for point in points:
                if clustered[point] == -1:
                    sub_clusters[i].append(point)
                    clustered[point] = i
            j += 1
    return sub_clusters, clustered

def calculate_shortest_distances_matrix(data, cluster):
    G = nx.Graph()
    G.add_nodes_from(cluster)
    i=0
    j=0
    for i in range(len(cluster)):
        for j in range(i+1,len(cluster)):
            dist = np.linalg.norm(data[cluster[i]] - data[cluster[j]])
            G.add_edge(cluster[i], cluster[j], weight=dist)
    MST = nx.minimum_spanning_tree(G)
    shortest_distances_matrix = nx.floyd_warshall_numpy(MST, weight='weight')
    return shortest_distances_matrix


def cal_compactness(data, cluster):
    cluster_data = data[cluster]
    centroid = np.mean(cluster_data, axis=0)
    dist_sum = 0
    for i in range(len(cluster_data)):
        dist_sum += np.sqrt(np.sum((cluster_data[i] - centroid) ** 2))
    compactness = dist_sum / len(cluster_data)
    return compactness


def divide_sub_clusters(data, sub_clusters, dist,lambda_ratio,belta):
    i=0
    while 1:
        if i==len(sub_clusters):
            break
        cluster=sub_clusters[i]
        matrix=calculate_shortest_distances_matrix(data, cluster)
        max_index = np.argmax(matrix)
        max_row, max_col = np.unravel_index(max_index, matrix.shape)
        row_index, col_index = sub_clusters[i][max_row], sub_clusters[i][max_col]
        ratio = matrix[max_row, max_col] / dist[row_index, col_index]
        if ratio >= lambda_ratio and len(sub_clusters[i])>=belta:
            new_cluster1 = [int(row_index)]
            new_cluster2 = [int(col_index)]
            for point in cluster:
                if point not in new_cluster1 and point not in new_cluster2:
                    dist_to_new_cluster1 = dist[point, new_cluster1[0]]
                    dist_to_new_cluster2 = dist[point, new_cluster2[0]]
                    if dist_to_new_cluster1 < dist_to_new_cluster2:
                        new_cluster1.append(point)
                    else:
                        new_cluster2.append(point)
            comp = cal_compactness(data, cluster)
            comp1 = cal_compactness(data, new_cluster1)
            comp2 = cal_compactness(data, new_cluster2)
            comp_w = (comp1 * len(new_cluster1) + comp2 * len(new_cluster2)) / len(cluster)
            if comp_w <= comp:
                sub_clusters.append(new_cluster1)
                sub_clusters.append(new_cluster2)
                sub_clusters.remove(cluster)
            else:
                i+=1
                continue
        else:
            i += 1
            continue
    return sub_clusters


def get_shared_nearest_neighbors(cluster1, cluster2, nbrs):
    neighbors1 = set()
    neighbors2 = set()
    for point in cluster1:
        neighbors1.update(nbrs[point])
    for point in cluster2:
        neighbors2.update(nbrs[point])
    snn = len(neighbors1.intersection(neighbors2))
    return snn

def filter_noise(sub_clusters):
    new_sub_clusters = []
    noise = []
    for sub_cluster in sub_clusters:
        if len(sub_cluster) >= 2:
            new_sub_clusters.append(list(sub_cluster))
        else:
            for p in sub_cluster:
                noise.append(p)
    return new_sub_clusters, noise

def calculate_cluster_centroids(data, sub_clusters):
    centroids = []
    for cluster in sub_clusters:
        cluster_data = data[cluster]
        centroid = np.mean(cluster_data, axis=0)
        centroids.append(centroid)
    return np.array(centroids)


def get_similarity_d(data, sub_clusters, nbrs, dist, rhos):
    m = len(sub_clusters)
    centroids = calculate_cluster_centroids(data, sub_clusters)
    c_dist = squareform(pdist(centroids))
    similarity_matrix = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            cluster1 = sub_clusters[i]
            cluster2 = sub_clusters[j]
            snn = get_shared_nearest_neighbors(cluster1, cluster2, nbrs)
            similarity = snn / (1 + c_dist[i, j])
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    return similarity_matrix


def get_point_labels(data, sub_clusters, cluster_labels):
    point_labels = np.full(len(data), -1)
    for cluster_idx, cluster in enumerate(sub_clusters):
        for point in cluster:
            point_labels[point] = cluster_labels[cluster_idx]
    return point_labels

def get_noise_label(noise, nbrs, label):
    for p in noise:
        for j in nbrs[p]:
            if j not in noise:
                label[p] = label[j]
                break
    return label

def MDMSC(data,y,n_clusters,k,dist,lambda_ratio,belta):
    rho, nbrs_dist, nbrs = ger_rho(data, k)
    core = find_core(rho, nbrs)
    delta, leader = find_leader(data, rho, nbrs, core, dist)
    sub_clusters, clustered = get_sub_cluster(data, core, leader)
    new_sub_clusters= divide_sub_clusters(data, sub_clusters, dist,lambda_ratio,belta)
    new_sub_clusters, noise = filter_noise(new_sub_clusters)
    similarity_matrix = get_similarity_d(data, new_sub_clusters, nbrs, dist,rho)
    similarity_matrix = similarity_matrix / np.max(similarity_matrix)
    for j in range(similarity_matrix.shape[0]):
        similarity_matrix[j,j]=1
    degree_matrix = np.diag(similarity_matrix.sum(axis=1))
    laplacian = degree_matrix - similarity_matrix
    norm_laplacian = np.matmul(np.linalg.inv(np.sqrt(degree_matrix)),
                               np.matmul(laplacian, np.linalg.inv(np.sqrt(degree_matrix))))
    eigenvalues, eigenvectors = np.linalg.eigh(norm_laplacian)

    features = eigenvectors[:, :n_clusters]

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels= kmeans.fit_predict(features)

    label = get_point_labels(data,new_sub_clusters, cluster_labels)
    label=get_noise_label(noise,nbrs,label)

    return label
