import matplotlib.pyplot as plt
import random
import pandas as pd
import math
import argparse
parser = argparse.ArgumentParser()

class KNN:
    def __init__(self, k, ax):
        """
        Parameters:
            k (int): number of clusters
            ax (axes): matplotlib axes to plot the clusters and the prediction
        """
        self.k = k
        self.ax = ax
    
    def fit(self, x, y):
        """
        Training the model
        
        Parameters:
            x (list): list of points in the training set
            y (list): corresponding labels for each cluster
        """
        self.x = x
        self.y = y
    
    def plot(self, point, label):
        """
        Plots the clusters
        
        Parameters:
            point (list): point to be classified
            label (list): predicted label
        """
        self.ax.scatter([point[0]], [point[1]], color='y')
        plt.show()
        self.ax.scatter([100], [100], color='y')
        plt.show()

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


def main(point):
    # Create training set
    k = 5
    cluster_spaces = [
        [(0,5), (0,5)],
        [(10,12), (14,17)],
        [(100,120), (10,13)],
        [(-10,-20), (-30,-35)],
        [(20,23), (14,17)]
    ]
    x_train, y_train = [], []
    for i in range(k):
        n = random.randint(20, 50)
        x = [(random.uniform(*cluster_spaces[i][0]), random.uniform(*cluster_spaces[i][1])) for _ in range(n)]
        y = ["__label__"+str(i) for _ in range(n)]
        x_train += x
        y_train += y
    train_df = pd.DataFrame({'x': x_train, 'y': y_train})
    
    # Scatter plot
    fig=plt.figure()
    ax=fig.add_axes([0,0,1,1])
    colors = "bgrcm"
    for i in range(k):
        curr_label_x = list(train_df[train_df['y']=="__label__"+str(i)]['x'].values)
        ax.scatter([point[0] for point in curr_label_x], [point[1] for point in curr_label_x], color=colors[i])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Clusters')
    
    # Creating knn object, predicting and plotting the clusters
    knn = KNN(k, ax)
    knn.fit(x_train, y_train)
    label = knn.predict(point)
    print("Predicted label", label)
    knn.plot(point, label)

if __name__ == "__main__":
    parser.add_argument("--point", nargs='+', help="point to be classified", required=True)
    args = parser.parse_args()
    point = args.point
    point = [float(i) for i in point]
    main(point)
