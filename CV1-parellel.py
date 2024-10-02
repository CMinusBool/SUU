import os
import pandas as pd
import numpy as np
import time
from joblib import Parallel, delayed


# function for generating all combinations of up to specified length
def all_combinations(elements, length):
    result = []
    combinations_generator(elements, [], length, result)
    return result


# help function for generating combinations
def combinations_generator(elements, current_combination, length, result):
    if len(current_combination) == length:
        return
    for i in range(len(elements)):
        new_combination = current_combination + [elements[i]]
        result.append(new_combination)
        combinations_generator(elements[i + 1:], new_combination, length, result)


def count_single_combination(combination, data):
    combination_set = set(combination)
    return sum(1 for dp in data if combination_set.issubset(dp))

####################################################################
#------------------------------CONFIG-------------------------------
####################################################################
combinations_len = 3
support_threshold = 0.25
min_confidence = 0.5
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'connect.dat')
####################################################################
#------------------------------CONFIG-------------------------------
####################################################################
start_time = time.time()
# Read the file and find the largest element (for generating combinations)
data = []
max_element = 0
data_point_count=0
with open(file_path, 'r') as file:
    for line in file:
        line=[int(x) for x in line.strip().split(' ')]
        if max(line)>max_element:
            max_element=max(line)
        data.append(line)
        data_point_count += 1

# generate all possible combinations
combinations = all_combinations(np.arange(1, max_element+1), combinations_len)

# for each possible combination check how many times it is present in dataset
# using joblib
counts = Parallel(n_jobs=-1)(delayed(count_single_combination)(comb, data) for comb in combinations)
combinations_tuples = [tuple(comb) for comb in combinations]
present_combinations = dict(zip(combinations_tuples, counts))

# calculate support for each present combination
combinations_support={}
print(f"Support for present combinations higher than minimal support:")
for combination, occurrence in present_combinations.items():
    support=occurrence/data_point_count
    if support > support_threshold:
        combinations_support[combination] = support
        print(f"Combination: {combination}, Support: {occurrence/data_point_count}")

# calculate confidence
print(f"Confidence for present combinations higher than minimal confidence:")
for combination, support in combinations_support.items():
    combination_array = np.array(combination)
    if len(combination_array) > 1:  # Check if there is any confidence to calculate
        combinations = all_combinations(combination_array, len(combination_array)-1)  # generate combinations to check
        # calculate confidence for elements that are missing from combination
        for c in combinations:
            confidence = combinations_support[tuple(combination)] / combinations_support[tuple(c)]
            if confidence > min_confidence:
                supplement = np.setdiff1d(combination_array, c)
                print(f"{c} -> {supplement} conf = {confidence}")

end_time = time.time()
print(f"Time to finish {end_time - start_time}s")