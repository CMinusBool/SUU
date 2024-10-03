import os
import pandas as pd
import numpy as np
import time
from joblib import Parallel, delayed


# calculate manhattan distances from each point to all other points
def manhattan_dis(data):
    distances = [[0] * len(data) for _ in range(len(data))]

    for i in range(len(data)):
        for y in range(i, len(data)):
            # calculates each distance only once and puts it to both row and col
            distances[i][y] = abs(data[i][0] - data[y][0]) + abs(data[i][1] - data[y][1])
            distances[y][i] = abs(data[i][0] - data[y][0]) + abs(data[i][1] - data[y][1])
    return distances


# common function
def hierarchical_clustering(data, single = True, manhattan = True, target):
    return manhattan_dis(data)


####################################################################
#------------------------------CONFIG-------------------------------
####################################################################
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'clusters3.csv')
####################################################################
#------------------------------CONFIG-------------------------------
####################################################################
start_time = time.time()
# Read the file
data = []
with open(file_path, 'r') as file:
    for line in file:
        line=[float(x) for x in line.strip().split(';')]
        data.append(line)


distances = hierarchical_clustering(data, True, True)
end_time = time.time()
print(f"Time to finish {end_time - start_time}s")
