import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer
import sklearn.cluster
import time
import pandas as pd


def jaccard_similarity(list1, list2):
    intersection_size = len(set(list1).intersection(set(list2)))
    union_size = len(set(list1).union(set(list2)))
    return intersection_size / union_size if union_size > 0 else 0

def intra_cluster_similarity(clusters):
    similarities = []

    for cluster in clusters:
        num_samples = len(cluster)
        similarity_matrix = np.zeros((num_samples, num_samples))

        for i in range(num_samples):
            for j in range(i + 1, num_samples):
                similarity_matrix[i, j] = jaccard_similarity(cluster[i], cluster[j])
                similarity_matrix[j, i] = similarity_matrix[i, j]

        sum_similarity = similarity_matrix.sum()
        similarities.append(sum_similarity)

    return similarities

def ICS(clusters):
    similarities = intra_cluster_similarity(clusters)
    s=0
    for i, similarity in enumerate(similarities):
        s+=similarity
    return s*0.5

def benchmark_algorithm(data, cluster_function, function_args=[], function_kwds={},
                        max_time=120, sample_size=5):
    
    result = np.nan * np.ones(sample_size)
    clusters_list = []
    
    for s in range(sample_size):
        mlb = MultiLabelBinarizer()
        one_hot_encoded_data = mlb.fit_transform(data)
        
        start_time = time.time()
        cluster_labels = cluster_function.fit_predict(one_hot_encoded_data, *function_args, **function_kwds)
        time_taken = time.time() - start_time
        
        if time_taken > max_time:
            result[s] = time_taken
            return pd.DataFrame({'x': [len(data)] * sample_size, 'y': result}), clusters_list
        else:
            result[s] = time_taken
            
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(data[i])
            clusters_list.append(list(clusters.values()))
        
    return pd.DataFrame({'x': [len(data)] * sample_size, 'y': result}), clusters_list

# datasets



transaction_data =[]
#path='dataset/tictactoefinal.txt'
path='dataset/zoofinal.txt'
#path='dataset/votefinal.txt'
#path='dataset/soybeanfinal.txt'
#path='dataset/primaryTumorfinal.txt'
#path='dataset/mushroomfinal.txt'
#path='dataset/lymphfinal.txt'
with open(path,'r') as file:
    for line in file:
        et=line.split(" ")
        del et[-1]
        transaction_data.append([int (ee) for ee in et])

start_time = time.time()

#clusterer = sklearn.cluster.KMeans(n_clusters=30)
#clusterer = sklearn.cluster.Birch(n_clusters=30)
#clusterer = sklearn.cluster.SpectralClustering(n_clusters=30)
clusterer = sklearn.cluster.AgglomerativeClustering(n_clusters=30)

results, clusters = benchmark_algorithm(transaction_data, clusterer)

end_time=time.time()
s=0
for e in clusters[0]:
    s=s+len(e)
print("detected transactions into clusters: ",s)
print("Number of  clusters found: ",len(clusters[0]))
print("Run-Time: " ,end_time-start_time)
print("ICS: ",ICS(clusters[0]))

