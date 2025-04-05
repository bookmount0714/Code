
import numpy as np
import torch
import subprocess
import umap.umap_ as umap
import hdbscan
from modules import globals_ele
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
def dimension_reduce_report(x):
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(x)
    return embedding


reports_tensor = globals_ele.reports_tensor

embedding = dimension_reduce_report(reports_tensor)

clusterer = hdbscan.HDBSCAN(min_cluster_size=15,metric='euclidean')
cluster_labels = clusterer.fit_predict(embedding)

# 处理噪声点，将 -1 替换为聚类数目
num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
cluster_labels = np.where(cluster_labels == -1, num_clusters, cluster_labels)

# 将标签转换为独热编码
encoder = OneHotEncoder(sparse=False)
one_hot_encoded_labels = encoder.fit_transform(cluster_labels.reshape(-1, 1))

# 将结果转换为 DataFrame 以便查看
one_hot_encoded_df = pd.DataFrame(one_hot_encoded_labels, columns=[f'Cluster_{i}' for i in range(one_hot_encoded_labels.shape[1])])
print(one_hot_encoded_df.head())

print(cluster_labels)