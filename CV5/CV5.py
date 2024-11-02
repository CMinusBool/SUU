import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# function to calculate sum of gini index of each split
def gini(splits):
    total_gini = 0
    #TODO in split is not what i expected
    for split in splits:
        n = len(split)
        if n == 0:
            continue
        # count number of each class in split
        counts = split[split.columns[-1]].value_counts()
        gini = 1
        for count in counts:
            gini -= (count / n) ** 2
        total_gini += gini * n / len(splits)

    return total_gini



# Function fo spliting dataset
def split_dataset(dataset, feature, value):
    left = dataset[dataset[feature] <= value]
    right = dataset[dataset[feature] > value]

    return left, right


def train_decision_tree(dataset, max_depth=0):
    #TODO: while loop for max_depth or until gini = 0
    classes = dataset[dataset.columns[-1]].unique()
    for feature in dataset.columns[:-1]:
        print(f'Feature: {feature}')
        best_gini = 1
        for value in dataset[feature]:
            left, right = split_dataset(dataset, feature, value)
            gini_coef = gini([left, right])
            print(f'Value: {value}, Gini: {gini_coef}')
            if gini_coef < best_gini:
                best_gini = gini_coef
                best_feature = feature
                best_value = value
                left_branch = left
                right_branch = right
        print(f'Best feature: {best_feature}, Best value: {best_value}, Best gini: {best_gini}')

#TODO  trained decision tree fucntin on test data

# Load the data
df = pd.read_csv('data_classification/sep.csv', header=None, delimiter=';')

### Split the data into training and test sets

X = df.drop(columns=df.columns[-1])
y = df[df.columns[-1]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

print(X_train.shape, X_test.shape)

# test code
print("training decision tree")
train_decision_tree(df)

#TODO: test decision tree on test data
