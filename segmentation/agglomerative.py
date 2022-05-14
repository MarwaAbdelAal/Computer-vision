import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
np.random.seed(42)


# KMeans Algorithm
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def clusters_distance(cluster1, cluster2):
    """
    Computes distance between two centroids of the two clusters
    cluster1 and cluster2 are lists of lists of points
    """
    cluster1_center = np.average(cluster1, axis=0)
    cluster2_center = np.average(cluster2, axis=0)
    return euclidean_distance(cluster1_center, cluster2_center)


class AgglomerativeClustering:
    def __init__(self, source: np.ndarray, clusters_numbers: int = 2, initial_k: int = 25):
        self.clusters_num = clusters_numbers
        self.initial_k = initial_k
        src = np.copy(source.reshape((-1, 3)))

        self.fit(src)

        self.output_image = [[self.predict_center(list(src)) for src in row] for row in source]
        self.output_image = np.array(self.output_image, np.uint8)


    def initial_clusters(self, points):
        """
        partition pixels into self.initial_k groups based on color similarity
        """
        groups = {}
        d = int(256 / self.initial_k)
        for i in range(self.initial_k):
            j = i * d
            groups[(j, j, j)] = []
        for i, p in enumerate(points):
            if i % 100000 == 0:
                print('processing pixel:', i)
            go = min(groups.keys(), key=lambda c: euclidean_distance(p, c))
            groups[go].append(p)
        return [g for g in groups.values() if len(g) > 0]


    def fit(self, points):
        # initially, assign each point to a distinct cluster
        print('Computing initial clusters ...')
        self.clusters_list = self.initial_clusters(points)
        print('number of initial clusters:', len(self.clusters_list))
        print('merging clusters ...')

        while len(self.clusters_list) > self.clusters_num:
            # Find the closest (most similar) pair of clusters
            cluster1, cluster2 = min(
                [(c1, c2) for i, c1 in enumerate(self.clusters_list) for c2 in self.clusters_list[:i]],
                key=lambda c: clusters_distance(c[0], c[1]))

            # Remove the two clusters from the clusters list
            self.clusters_list = [c for c in self.clusters_list if c != cluster1 and c != cluster2]

            # Merge the two clusters
            merged_cluster = cluster1 + cluster2

            # Add the merged cluster to the clusters list
            self.clusters_list.append(merged_cluster)

            print('number of clusters:', len(self.clusters_list))

        print('assigning cluster num to each point ...')
        self.cluster = {}
        for cl_num, cl in enumerate(self.clusters_list):
            for point in cl:
                self.cluster[tuple(point)] = cl_num

        print('Computing cluster centers ...')
        self.centers = {}
        for cl_num, cl in enumerate(self.clusters_list):
            self.centers[cl_num] = np.average(cl, axis=0)


    def predict_cluster(self, point):
        """
        Find cluster number of point
        """
        # assuming point belongs to clusters that were computed by fit functions
        return self.cluster[tuple(point)]


    def predict_center(self, point):
        """
        Find center of the cluster that point belongs to
        """
        point_cluster_num = self.predict_cluster(point)
        center = self.centers[point_cluster_num]
        return center


def apply_agglomerative(source: np.ndarray, clusters_numbers: int = 2, initial_clusters: int = 25):

    src = np.copy(source)
    color_img = cv2.cvtColor(src, cv2.COLOR_Luv2BGR)
    
    agglomerative = AgglomerativeClustering(source=color_img, clusters_numbers=clusters_numbers,
                                            initial_k=initial_clusters)

    return agglomerative.output_image


if __name__ == "__main__":
    img = cv2.imread('images/landscape.png')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    start_time = time.time()
    binaryImg = apply_agglomerative(img)
    end_time = time.time()

    regionGrowing_time = format(end_time - start_time, '.5f')
    print(f'Region Growing Computation Time = {regionGrowing_time} sec')    # print(  end_time - start_time)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes.ravel()
    
    ax[0].imshow(img_rgb)
    ax[0].set_title('Original Image')
    ax[0].set_axis_off()
    
    ax[1].imshow(binaryImg ,cmap = "gray")
    ax[1].set_title('LUV Image')
    ax[1].set_axis_off()
    
    plt.tight_layout()
    plt.show()