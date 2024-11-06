import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from anyio.abc import value
from debugpy.common.timestamp import current
from pandas.io.sas.sas_constants import dataset_length
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# function to calculate sum of gini index of each split
def gini(splits):
    total_gini = 0
    dataset_length = sum([len(split) for split in splits])
    individual_gini = []
    for split in splits:
        n = len(split)
        if n == 0:
            individual_gini.append(0)
        # count number of each class in split
        counts = np.unique(split[:, -1], return_counts=True)[1]
        gini = 1
        for count in counts:
            gini -= (count / n) ** 2
        individual_gini.append(gini)
        total_gini += gini * (n / dataset_length) # weigh the gini index by the number of samples in the split

    return total_gini, individual_gini

def find_best_split_in_node(dataset):
    if dataset is None or len(dataset) == 0:
        return None
    best_gini = 1
    for feature in range(len(dataset[0])-1): # last column is class
        for row in dataset:
            value = row[feature]
            left, right = split_dataset(dataset, feature, value)
            gini_coef, individual_gini = gini([left, right])
            if gini_coef < best_gini:
                best_gini = gini_coef
                best_feature = feature
                best_value = value
                left_branch = left
                right_branch = right
                left_gini = individual_gini[0]
                right_gini = individual_gini[1]

    #print(f'Best feature: {best_feature}, Best value: {best_value}, Best gini: {best_gini}')

    return best_feature, best_value, best_gini, left_branch, right_branch, left_gini, right_gini

def decide_class(split):
    classes, counts = np.unique(split[:, -1], return_counts=True)
    return classes[np.argmax(counts)]

def find_recursively_best_splits(dataset, max_depth, current_depth):
    best_feature, best_value, best_gini, left_branch, right_branch, left_gini, right_gini = find_best_split_in_node(dataset)

    # check if leaf was reached
    left_leaf = False
    right_leaf = False
    if left_gini == 0:
        left_leaf = True
    if right_gini == 0:
        right_leaf = True

    left_tree = None
    right_tree = None
    left_class = None
    right_class = None
    if current_depth != max_depth:
        if not left_leaf:
            left_tree = find_recursively_best_splits(left_branch, max_depth, current_depth+1)
        else:
            left_class = decide_class(left_branch)
        if not right_leaf:
            right_tree = find_recursively_best_splits(right_branch, max_depth, current_depth+1)
        else:
            right_class = decide_class(right_branch)

    else:# if we are at max depth we decide the class
        left_class = decide_class(left_branch)
        right_class = decide_class(right_branch)

    return {'feature': best_feature, 'value': best_value, 'gini': best_gini, 'left_tree': left_tree,
            'right_tree': right_tree, 'left_class': left_class, 'right_class': right_class}

# Function fo spliting dataset
def split_dataset(dataset, feature, value):
    dataset = np.array(dataset)  # Ensure the dataset is a NumPy array
    left = dataset[dataset[:, feature] < value]
    right = dataset[dataset[:, feature] >= value]
    return left, right


def train_decision_tree(dataset, max_depth=-1):
    #convert dataset to array
    dataset = dataset.to_numpy()

    # recursive function to build the tree
    tree = find_recursively_best_splits(dataset, max_depth, 1)

    return tree


def print_decision_tree(tree, indent=""):
    # Check if we are at a leaf node
    if 'left_tree' not in tree and 'right_tree' not in tree:
        if 'left_class' in tree:
            print(indent + f"Left: -Leaf- Class {tree['left_class']}")
        elif 'right_class' in tree:
            print(indent + f"Right: -Leaf- Class {tree['right_class']}")
        return

    # Print the current decision node's information
    print(f"{indent}Feature: {tree['feature']}, Value: {tree['value']}, Gini: {tree['gini']}")

    # Print the left subtree or leaf class if it's a leaf
    if tree.get('left_tree') is not None:
        print(indent + "Left:")
        print_decision_tree(tree['left_tree'], indent + "    ")
    elif 'left_class' in tree:
        print(indent + f"Leaf: Class {tree['left_class']}")

    # Print the right subtree or leaf class if it's a leaf
    if tree.get('right_tree') is not None:
        print(indent + "Right:")
        print_decision_tree(tree['right_tree'], indent + "    ")
    elif 'right_class' in tree:
        print(indent + f"Leaf: Class {tree['right_class']}")

# asign class to a sample acoriding to trained tree
def asign_class(tree, sample):
    go_left = False

    if sample[tree['feature']] < tree['value']:
        go_left = True

    if go_left:
        if tree['left_tree'] is not None:
            return asign_class(tree['left_tree'], sample)
        else:
            return tree['left_class']
    else:
        if tree['right_tree'] is not None:
            return asign_class(tree['right_tree'], sample)
        else:
            return tree['right_class']

def run_decision_tree(tree, dataset):
    dataset = dataset.to_numpy()
    #remove last column
    dataset = dataset[:, :-1]
    predictions = []

    #predict the class of each sample
    for row in dataset:
        predictions.append(asign_class(tree, row))

    return predictions

# Load the data
df = pd.read_csv('data_classification/iris.csv', header=None, delimiter=';')

#Split the data into training and test sets

X = df.drop(columns=df.columns[-1])
y = df[df.columns[-1]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

Xy_train = pd.concat([X_train, y_train], axis=1)
Xy_test = pd.concat([X_test, y_test], axis=1)


# test code
print("training decision tree for iris dataset")
tree = train_decision_tree(Xy_train)
# Print the decision tree
#print_decision_tree(tree)

print("running decision tree")
predictions = run_decision_tree(tree, Xy_test)

# Calculate accuracy
correct = 0
for i in range(len(predictions)):
    if predictions[i] == y_test.iloc[i]:
        correct += 1

accuracy = correct / len(predictions)
print(f"Accuracy from decision tree: {accuracy}")

# Compare with sklearn

# Create a decision tree classifier
clf = DecisionTreeClassifier(criterion='gini', max_depth=20)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)

# Calculate accuracy
correct = 0
for i in range(len(predictions)):
    if predictions[i] == y_test.iloc[i]:
        correct += 1

accuracy = correct / len(predictions)
print(f"Accuracy from sklearn: {accuracy}")

print("Testing the max depth parametere from 1 to 15 on titanic dataset")
# Load the data
df = pd.read_csv('data_classification/titanic_preprocessed.csv', delimiter=',')

#Split the data into training and test sets. class is the second column

X = df.drop(columns=df.columns[1])
y = df[df.columns[1]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

Xy_train = pd.concat([X_train, y_train], axis=1)
Xy_test = pd.concat([X_test, y_test], axis=1)


accuracies = []
train_accuracies = []

for max_depth in range(1, 16):
    print(f"Max depth: {max_depth}")
    # train decision tree
    tree = train_decision_tree(Xy_train, max_depth)

    # run decision tree
    predictions = run_decision_tree(tree, Xy_test)

    # Calculate accuracy
    correct = 0
    for i in range(len(predictions)):
        if predictions[i] == y_test.iloc[i]:
            correct += 1

    accuracy = correct / len(predictions)
    print(f"Accuracy from my decision tree: {accuracy}")

    #acuracy in training
    training_predictions = run_decision_tree(tree, Xy_train)
    correct = 0
    for i in range(len(training_predictions)):
        if training_predictions[i] == y_train.iloc[i]:
            correct += 1

    training_accuracy = correct / len(training_predictions)
    print(f"Accuracy from my decision tree in training: {training_accuracy}")

    accuracies.append(accuracy)
    train_accuracies.append(training_accuracy)

# without max depth
print(f"Max depth: -1")
# train decision tree
tree = train_decision_tree(Xy_train)
# run decision tree
predictions = run_decision_tree(tree, Xy_test)

# Calculate accuracy
correct = 0
for i in range(len(predictions)):
    if predictions[i] == y_test.iloc[i]:
        correct += 1

accuracy = correct / len(predictions)
print(f"Accuracy from my decision tree: {accuracy}")
accuracies.append(accuracy)

#acuracy in training
training_predictions = run_decision_tree(tree, Xy_train)
correct = 0
for i in range(len(training_predictions)):
    if training_predictions[i] == y_train.iloc[i]:
        correct += 1

training_accuracy = correct / len(training_predictions)
print(f"Accuracy from my decision tree in training: {training_accuracy}")
train_accuracies.append(training_accuracy)

plt.plot(range(1, 17), accuracies)
plt.plot(range(1, 17), train_accuracies)
plt.xlabel('Max depth')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Max depth')
xticks = list(range(1, 17))
xtick_labels = [str(x) for x in xticks]
xtick_labels[-1] = 'inf'
plt.xticks(xticks, xtick_labels)

plt.legend(['Test', 'Train'])
plt.show()

# conclusions
# accuracy in training keeps increasing with depth up to 100% accuracy
# what exact depth is the best changes even on same dataset depending on test size and
# randomness of the split
# for 20/80 split and random state 42
# in depth 1 and 2 there is underfiting - The tree is too simple
# in depth 3 the accuracy is the same as in depth 3 - best depth
# in depths higher than 4 the accuracy is lower - overfitting




