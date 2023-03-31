#!/usr/bin/env python3

'''
    Author: Stiven LaVrenov
    somethin
    Description: Use a k-means clustering algorithm to train on a dataset, and return test predictions
    Usage: ./kmeans.py [# of Clusters] [Training Data] [Test Data]
'''

import sys
import numpy as np

if len(sys.argv) != 4:
    print('\n', f'Usage: {sys.argv[0]} [# of clusters] [training data] [validation data]', '\n')
    sys.exit(1)

# Class method for K-Means Algorithm
class KMeans():
    # Initialize the class with the number of centers/centroids
    def __init__(self, centers):
        self.centers = centers

    # Use the first k number of training examples as centroids, and keep track of labels
    def initialize_centroids(self, data):
        centroids = data[:self.centers, 0:-1]
        labels = data[:self.centers, -1]

        return centroids, labels

    # Calculate the euclidian distance between each point and each centroid
    def euclidian(self, centroids, points):
        distances = [np.sqrt(np.sum((points - centroids[x]) ** 2)) for x in range(self.centers)]
        return np.argmin(distances)

    # Assign each centroid index the data and label of each corresponding data point that it's closest to
    def compute_centroids(self, centroids, data):
        centroid = [[] for _ in range(self.centers)]
        centroid_label = [[] for _ in range(self.centers)]

        for x in range(len(data[self.centers:])):
            c_id = self.euclidian(centroids, data[x+self.centers,:-1])
            centroid[c_id].append(data[x+self.centers,:-1])
            centroid_label[c_id].append(data[x+self.centers,-1])
        return centroid, centroid_label

    # Get the new centroid locations
    def recompute_centroids(self, centroid, centroid_label, c_labels):
        centroid_labels = [[] for _ in range(self.centers)]
        for c, c_id in enumerate(centroid_label):
            unique, count = np.unique(c_id, return_counts=True)
            if np.isnan(count).all() == False:
                centroid_labels[c] = np.max(c_id)
        for c, label in enumerate(centroid_labels):
            if np.isnan(label).any():
                centroid_labels[c] = int(c_labels(c))
        return [np.mean(cen_id, axis = 0) for cen_id in centroid], centroid_labels

    # Determine how many test data points were correctly predicted
    def predict(self, data):
        correct = 0
        actual = data[:,-1]
        for x in range(len(data)):
            c_id = self.euclidian(self.centroids, data[x,:-1])
            if self.labels[c_id] == int(actual[x]):
                correct += 1
        return correct

    # Recursively go through the k-means algorithm until no new centroid locations are found
    def fit(self, data):
        self.centroids, self.labels = self.initialize_centroids(data)
        self.previous_centroid = None

        while np.not_equal(self.centroids, self.previous_centroid).all():
            self.centroid, self.label = self.compute_centroids(self.centroids, data)
            self.previous_centroid = self.centroids
            self.centroids, self.labels = self.recompute_centroids(self.centroid, self.label, self.labels)
            for c, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():
                    self.centroids[c] = self.previous_centroid[c]

# Load and validates data for ease
def load_data(file):
    data = np.loadtxt(file)
    if len(data.shape) < 2:
        data = np.array([data])
    return data

def main():
    # Number of centers, or clusters, to separate classification into
    centers = int(sys.argv[1])

    # Load training data
    train = load_data(sys.argv[2])

    # Load test data to predict against clusters
    test = load_data(sys.argv[3])

    # Initialize KMeans Class and fit the training data
    kmeans = KMeans(centers)
    kmeans.fit(train)

    # Determine the test results
    test_results = kmeans.predict(test)
    print(test_results)

if __name__ == '__main__':
    main()