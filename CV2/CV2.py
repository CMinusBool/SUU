import math
import os
from enum import Enum

import pandas as pd
import numpy as np
import time
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from itertools import cycle
from scipy.cluster.hierarchy import dendrogram


class DistanceMetric(Enum):
    MANHATTAN = 'manhattan'
    EUCLIDEAN = 'euclidean'
    COSINE = 'cosine'


class LinkageType(Enum):
    SINGLE = 'single'
    COMPLETE = 'complete'

def min_max_normalize(data):
    maximum_x = -math.inf
    maximum_y = -math.inf
    minimum_x = math.inf
    minimum_y = math.inf
    for i in range(len(data)):
        if data[i][0] > maximum_x:
            maximum_x = data[i][0]
        if data[i][0] < minimum_x:
            minimum_x = data[i][0]
        if data[i][1] > maximum_y:
            maximum_y = data[i][1]
        if data[i][1] < minimum_y:
            minimum_y = data[i][1]

    for i in range(len(data)):
        data[i][0] = (data[i][0] - minimum_x) / (maximum_x - minimum_x)
        data[i][1] = (data[i][1] - minimum_y) / (maximum_y - minimum_y)

    return data
def min_with_index(array):
    maximum = (math.inf, -1)
    for i in range(len(array)):
        if array[i] < maximum[0] and array[i] != 0:
            maximum = (array[i], i)
    return maximum

# returns minimum from double(comparing first element), returns also position in array
def min_double(array):
    minimum = (math.inf, -1, -1)
    for i in range(len(array)):
        if array[i][0] < minimum[0] and array[i][0] != 0:
            minimum = [array[i][0], array[i][1], i]
    return minimum
# calculate manhattan distances from each point to all other points
def manhattan_dis(data):
    distances = [[0] * len(data) for _ in range(len(data))]

    for i in range(len(data)):
        for y in range(i, len(data)):
            # calculates each distance only once and puts it to both row and col
            distance = abs(data[i][0] - data[y][0]) + abs(data[i][1] - data[y][1])
            distances[i][y] = distance
            distances[y][i] = distance
    return distances

def euclidean_dis(data):
    distances = [[0] * len(data) for _ in range(len(data))]

    for i in range(len(data)):
        for y in range(i, len(data)):
            # calculates each distance only once and puts it to both row and col
            distance = math.sqrt((data[i][0] - data[y][0])**2 + (data[i][1] - data[y][1])**2)
            distances[i][y] = distance
            distances[y][i] = distance
    return distances


def cosine_dis(data):
    distances = [[0] * len(data) for _ in range(len(data))]

    for i in range(len(data)):
        for y in range(i, len(data)):
            # calculates each distance only once and puts it to both row and col
            distance = np.dot(data[i], data[y]) / (np.linalg.norm(data[i]) * np.linalg.norm(data[y]))
            distances[i][y] = distance
            distances[y][i] = distance
    return distances
def complete_link_group(distances, elem1, elem2):
    # first group them
    distances[elem1][elem2] = 0
    distances[elem2][elem1] = 0
    # then put higher from 2 groups into new
    result = []
    for a, b in zip(distances[elem1], distances[elem2]):
        result.append(max(a, b)) # there aro no negative values so this works
    distances[elem1] = [0 * len(data) for _ in range(len(data))]
    distances[elem2] = result

    # update the values in other clusters - take max value from 2 old clusters
    for i in range(len(distances)):
        maximum = max(distances[i][elem1], distances[i][elem2])
        distances[i][elem1] = 0 # remove old group
        distances[i][elem2] = maximum
    return distances

#TODO single link

def complete_link(distances):
    min_distances = []
    for group in range(len(distances)):
        min_distances.append(min_with_index(distances[group]))
    min_distance = min_double(min_distances)
    # group points
    distances = complete_link_group(distances, min_distance[1], min_distance[2])
    return (distances, min_distance)


def single_link_group(distances, elem1, elem2):
    # first group them
    distances[elem1][elem2] = 0
    distances[elem2][elem1] = 0
    # then put smaller from 2 groups into new
    result = []
    for a, b in zip(distances[elem1], distances[elem2]):
        result.append(min(a, b))  # there aro no negative values so this works
    distances[elem1] = [0 * len(data) for _ in range(len(data))]
    distances[elem2] = result

    # update the values in other clusters - take min value from 2 old clusters
    for i in range(len(distances)):
        minimum = min(distances[i][elem1], distances[i][elem2])
        distances[i][elem1] = 0  # remove old group
        distances[i][elem2] = minimum
    return distances


def single_link(distances):
    min_distances = []
    for group in range(len(distances)):
        min_distances.append(min_with_index(distances[group]))
    min_distance = min_double(min_distances)
    # group points
    distances = single_link_group(distances, min_distance[1], min_distance[2])
    return (distances, min_distance)

# unify function
def hierarchical_clustering(data, linkage=LinkageType.COMPLETE,
                            distance_metric: DistanceMetric = DistanceMetric.MANHATTAN, target=1):
    groups = [[i] for i in range(len(data))]
    if distance_metric == DistanceMetric.MANHATTAN:
        distances = manhattan_dis(data)
    elif distance_metric == DistanceMetric.EUCLIDEAN:
        distances = euclidean_dis(data)
    elif distance_metric == DistanceMetric.COSINE:
        distances = cosine_dis(data)
    else:
        raise ValueError("Unknown distance metric")

    linkage_matrix = []

    # run corect number of iterations
    num_of_groups = len(distances)
    print("Data prepared")
    if linkage == LinkageType.COMPLETE:
        while(num_of_groups > target):
            distances, result = complete_link(distances)
            merged_group = groups[result[1]] + groups[result[2]]
            groups[result[1]] = []  # Clear old group
            groups[result[2]] = merged_group

            # Append to the linkage matrix: [cluster1, cluster2, distance]
            linkage_matrix.append([result[1], result[2], result[0]])
            num_of_groups -= 1
    elif linkage == LinkageType.SINGLE:
        while(num_of_groups > target):
            distances, result = single_link(distances)
            merged_group = groups[result[1]] + groups[result[2]]
            groups[result[1]] = []  # Clear old group
            groups[result[2]] = merged_group

            # Append to the linkage matrix: [cluster1, cluster2, distance]
            linkage_matrix.append([result[1], result[2], result[0]])
            num_of_groups -= 1

    return linkage_matrix, groups


def plot_clusters(data, groups, title="Clusters"):
    plt.figure(figsize=(8, 6))
    colors = cycle('bgrcmyk')  # Cycle through colors for each cluster
    cluster_num = 0
    for group, color in zip(groups, colors):
        if group:  # Only plot non-empty groups
            group_points = np.array([data[i] for i in group])
            plt.scatter(group_points[:, 0], group_points[:, 1], c=color, label=f'Cluster {cluster_num}')
            cluster_num += 1
    plt.title(title)
    plt.legend()
    plt.show()

def draw_dendrogram(linkage_matrix):
    # TODO - lines from middle of previous clusters
    plt.figure(figsize=(10, 7))
    # Draw the dendrogram manually
    # draw vertical lines at each points
    for i in range(len(linkage_matrix)):
        x = [linkage_matrix[i][0], linkage_matrix[i][1]]
        y = [linkage_matrix[i][2], linkage_matrix[i][2]]
        plt.plot(x, y, c='b')

    # draw horizontal line connecting two clusters
    for i in range(len(linkage_matrix)):
        x = [linkage_matrix[i][0], linkage_matrix[i][0]]
        y = [0, linkage_matrix[i][2]]
        plt.plot(x, y, c='b')
        x = [linkage_matrix[i][1], linkage_matrix[i][1]]
        y = [0, linkage_matrix[i][2]]
        plt.plot(x, y, c='b')

    plt.title("Dendrogram")
    plt.xlabel("Index of Data Points")
    plt.ylabel("Distance")
    plt.show()

####################################################################
#------------------------------CONFIG-------------------------------
####################################################################
script_dir = os.path.dirname(os.path.abspath(__file__))
file_name = 'clusters3.csv'
target = 3
file_path = os.path.join(script_dir, file_name)
####################################################################
#------------------------------CONFIG-------------------------------
####################################################################
start_time = time.time()
# Read the file
data = []
with open(file_path, 'r') as file:
    for line in file:
        line = [float(x) for x in line.strip().split(';')]
        data.append(line)

data = min_max_normalize(data)
linkage_matrix, groups = hierarchical_clustering(data, target=target, linkage=LinkageType.SINGLE,
                                                 distance_metric=DistanceMetric.EUCLIDEAN)
print("Calculations finished")

if target == 1:
    draw_dendrogram(linkage_matrix)
else:
    plot_clusters(data, groups, f"Clusters from file: '{file_name}'")

end_time = time.time()
print(f"Time to finish {end_time - start_time}s")

# TODO calculate only with lower triangle in distance matrix
# TODO - dont update so much in single linkage? idk
# TODO cosine similarity - larger value means more similar!
