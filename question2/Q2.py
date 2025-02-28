import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Entropy calculation function
def entropy(y):
    class_counts = y.value_counts()
    probabilities = class_counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

# Information gain calculation function
def information_gain(X, y, feature):
    total_entropy = entropy(y)
    values = X[feature].unique()
    weighted_entropy = sum((len(y[X[feature] == value]) / len(y)) * entropy(y[X[feature] == value]) for value in values)
    return total_entropy - weighted_entropy

# Decision tree node class
class Node:
    def __init__(self, feature=None, label=None, children=None):
        self.feature = feature
        self.label = label
        self.children = children if children else {}

# Build decision tree function
def build_tree(X, y, features):
    if len(y.unique()) == 1:
        return Node(label=y.iloc[0])
    if len(features) == 0:
        return Node(label=y.mode()[0])
    
    best_feature = max(features, key=lambda f: information_gain(X, y, f))
    tree = Node(feature=best_feature)
    remaining_features = [f for f in features if f != best_feature]
    
    for value in X[best_feature].unique():
        subset_X = X[X[best_feature] == value].drop(columns=[best_feature])
        subset_y = y[X[best_feature] == value]
        tree.children[value] = build_tree(subset_X, subset_y, remaining_features)
    
    return tree

# Convert tree to dictionary
def tree_to_dict(tree):
    if tree.label is not None:
        return {'label': tree.label}
    return {
        'feature': tree.feature,
        'children': {str(value): tree_to_dict(child) for value, child in tree.children.items()}
    }

# Predict function
def predict(tree, X):
    predictions = []
    for _, row in X.iterrows():
        node = tree
        while node.label is None:
            if row[node.feature] not in node.children:
                predictions.append(None)
                break
            node = node.children[row[node.feature]]
        else:
            predictions.append(node.label)
    return np.array(predictions)

# Evaluate model function
def evaluate(X, y, test_size=0.2, num_trials=100):
    train_accuracies = []
    test_accuracies = []
    
    for trial in range(1, num_trials + 1):
        print(f"Running trial {trial}...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=None)
        
        tree = build_tree(X_train, y_train, X.columns)
        
        train_predictions = predict(tree, X_train)
        test_predictions = predict(tree, X_test)
        
        train_acc = np.mean(train_predictions == y_train)
        test_acc = np.mean(test_predictions == y_test)
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        print(f"Trial {trial}: Train Accuracy = {train_acc:.4f}, Test Accuracy = {test_acc:.4f}")
        
        if trial == 1:
            tree_dict = tree_to_dict(tree)
            with open("decision_tree.json", 'w') as json_file:
                json.dump(tree_dict, json_file, indent=4)
            print("Decision tree saved as decision_tree.json")
    
    return train_accuracies, test_accuracies

# Load dataset
data = pd.read_csv('car.csv')
df = pd.DataFrame(data)
X = df.drop(columns=['class'])
y = df['class']

# Run evaluation
train_acc, test_acc = evaluate(X, y, test_size=0.2, num_trials=100)

# Plot histograms
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(train_acc, bins=10, edgecolor='black', alpha=0.7)
plt.xlabel('Training Accuracy')
plt.ylabel('Frequency')
plt.title('Training Accuracy Distribution')

plt.subplot(1, 2, 2)
plt.hist(test_acc, bins=10, edgecolor='black', alpha=0.7)
plt.xlabel('Testing Accuracy')
plt.ylabel('Frequency')
plt.title('Testing Accuracy Distribution')

plt.tight_layout()
plt.savefig('accuracy_histogram.png')

# Print mean and standard deviation
print(f"Mean Training Accuracy: {np.mean(train_acc):.4f} ± {np.std(train_acc):.4f}")
print(f"Mean Testing Accuracy: {np.mean(test_acc):.4f} ± {np.std(test_acc):.4f}")
