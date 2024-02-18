import numpy as np
import sklearn
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
if __name__ == '__main__':
    npy_file = '/media/yueyulin/KINGSTON/data/natural_questions_10_200_docs_q_p_n_tokenized_ds/neg_arr.npy'
    arr = np.load(npy_file)
    cluster_size = 10
    kmeans = KMeans(n_clusters=cluster_size, random_state=0).fit(arr.reshape(-1,1))
    print(kmeans.cluster_centers_)
    num_per_cluster = []
    for i in range(cluster_size):
        num_per_cluster.append(np.sum(kmeans.labels_==i))
    print(num_per_cluster, ' percentage: ', np.array(num_per_cluster)/np.sum(num_per_cluster))


    # 绘制数据点
    for i in range(cluster_size):
        plt.scatter(arr[kmeans.labels_ == i], np.zeros_like(arr[kmeans.labels_ == i]) + i)

    # 绘制簇中心
    plt.scatter(kmeans.cluster_centers_, range(cluster_size), color='red', marker='*')

    plt.show()