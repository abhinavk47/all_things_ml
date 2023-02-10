from sklearn.metrics import precision_score, recall_score, f1_score
import math
import random
import argparse
parser = argparse.ArgumentParser()

class Kmeans:
    def __init__(self, k):
        self.k = k
    
    def fit(self, x):
        """
        Creates clusters for the given training set

        Parameters:
            x (list): list of points
        
        Returns:
            list: list of points with clusters
        """
        centroids = random.sample(x, self.k)
        x = [[point, None] for point in x]
        converged = False
        idx = 0
        while not converged:
            error = 0
            idx += 1
            converged = True
            for point_w_label in x:
                point = point_w_label[0]
                distances = []
                for label, centroid in enumerate(centroids):
                    distance = math.sqrt((point[0]-centroid[0])**2 + (point[1]-centroid[1])**2)
                    distances.append((distance, label))
                min_distance, label = min(distances)
                if point_w_label[1] != label:
                    converged = False
                point_w_label[1] = label
            same_cluster_centroids = []
            for i in range(k):
                cluster_points = [point for point, label in x if label==i]
                if len(cluster_points) == 0:
                    same_cluster_centroids.append([i,i])
                    continue
                prev_centroid_value = centroids[i].copy()
                centroids[i][0] = sum([point[0] for point in cluster_points])/len(cluster_points)
                centroids[i][1] = sum([point[1] for point in cluster_points])/len(cluster_points)
                if abs(prev_centroid_value[0] - centroids[i][0]) > 0.1 or abs(prev_centroid_value[1] - centroids[i][1]) > 0.1:
                    converged = False
            for i in range(k):
                for j in range(i+1, k):
                    if math.sqrt((centroids[i][0]-centroids[j][0])**2 + (centroids[i][1]-centroids[j][1])**2) < 15:
                        same_cluster_centroids.append([i,j])
            print(centroids)
            print(same_cluster_centroids)
            for i, j in same_cluster_centroids:
                centroids[j] = random.choice(x)[0]
                converged = False
                break

        print("Number of iterations before convergence", idx)
        return x, centroids

class KNN:
    def __init__(self, k):
        """
        Parameters:
            k (int): number of clusters
        """
        self.k = k
    
    def fit(self, x, y):
        """
        Training the model
        
        Parameters:
            x (list): list of points in the training set
            y (list): corresponding labels for each cluster
        """
        self.x = x
        self.y = y
    
    def predict(self, point):
        """
        Predicts the cluster the point belongs to
        
        Parameters:
            point (list): point to be classified
        
        Returns:
            str: predicted cluster label
        """
        distances = []
        for X, label in zip(self.x, self.y):
            distance = math.sqrt((point[0] - X[0])**2 + (point[1] - X[1])**2)
            distances.append((distance, label))
        distances.sort()
        distances = distances[:self.k]
        labels = [label for _, label in distances]
        predicted_label = max(set(labels), key=labels.count)
        return predicted_label

def main(k):
    # Create training set
    num_clusters = k
    cluster_spaces = [
        [(0,5), (0,5)],
        [(50,52), (54,57)],
        [(100,120), (10,13)],
        [(-10,-20), (-30,-35)],
        [(220,223), (14,17)]
    ]
    x_train, y_train = [], []
    for i in range(len(cluster_spaces)):
        n = random.randint(120, 250)
        x = [[random.uniform(*cluster_spaces[i][0]), random.uniform(*cluster_spaces[i][1])] for _ in range(n)]
        y = [i for _ in range(n)]
        x_train += x
        y_train += y
    # Initiate knn object
    knn = KNN(num_clusters)
    knn.fit(x_train, y_train)
    # Initiate K means clustering object and set the clusters
    kmeans = Kmeans(k)
    X, centroids = kmeans.fit(x_train)
    label_mapping = {}
    for idx, centroid in enumerate(centroids):
        label = knn.predict(centroid)
        label_mapping[idx] = label
    print(label_mapping)
    predicted_labels = [label_mapping[label] for _, label in X]
    print("Precision", precision_score(y_train, predicted_labels, average=None))
    print("Recall", recall_score(y_train, predicted_labels, average=None))
    print("F1 score", f1_score(y_train, predicted_labels, average=None))

if __name__ == "__main__":
    parser.add_argument("--k", help="number of clusters", required=True)
    args = parser.parse_args()
    k = int(args.k)
    main(k)
