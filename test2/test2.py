import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Function to compute entropy
def entropy(y):
    class_counts = y.value_counts()
    probabilities = class_counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

# Function to compute information gain
def information_gain(X, y, feature):
    total_entropy = entropy(y)
    values = X[feature].unique()
    weighted_entropy = 0
    for value in values:
        subset_y = y[X[feature] == value]
        weighted_entropy += (len(subset_y) / len(y)) * entropy(subset_y)
    
    return total_entropy - weighted_entropy

# Class representing a decision tree node
class Node:
    def __init__(self, feature=None, label=None, children=None):
        self.feature = feature
        self.label = label
        self.children = children if children else {}

# Function to build the decision tree
def build_tree(X, y, features):
    if len(y.unique()) == 1:
        return Node(label=y.iloc[0])
    
    if len(features) == 0:
        return Node(label=y.mode()[0])
    
    best_feature = None
    best_info_gain = -np.inf
    for feature in features:
        info_gain = information_gain(X, y, feature)
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = feature
    
    tree = Node(feature=best_feature)
    remaining_features = [f for f in features if f != best_feature]
    
    for value in X[best_feature].unique():
        subset_X = X[X[best_feature] == value].drop(columns=[best_feature])
        subset_y = y[X[best_feature] == value]
        child_node = build_tree(subset_X, subset_y, remaining_features)
        tree.children[value] = child_node
    
    return tree

# Function to convert tree to dictionary (for JSON storage)
def tree_to_dict(tree):
    if tree.label is not None:
        return {'label': tree.label}
    
    return {
        'feature': tree.feature,
        'children': {str(value): tree_to_dict(child) for value, child in tree.children.items()}
    }

# Function to make predictions using the tree
def predict(tree, X):
    predictions = []
    for _, row in X.iterrows():
        node = tree
        while node.label is None:
            if row[node.feature] not in node.children:
                predictions.append(y.mode()[0])  # Assign most common class if unseen value
                break
            node = node.children[row[node.feature]]
        else:
            predictions.append(node.label)
    return np.array(predictions)

# Function to train and evaluate the decision tree
def evaluate(X, y, test_size=0.2, num_trials=100):
    train_accuracies = []
    test_accuracies = []
    
    for trial in range(num_trials):
        print(f"\n🔄 Trial {trial + 1} in progress...")  

        # Ensure randomness in splitting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=None, shuffle=True)
        print("✅ Data successfully split!")

        # Train the decision tree
        tree = build_tree(X_train, y_train, X.columns)
        print("🌲 Decision tree training completed!")

        # Save JSON tree structure only for the first trial
        if trial == 0:
            print("💾 Saving decision tree structure as 'decision_tree.json'...")
            with open("decision_tree.json", "w") as json_file:
                json.dump(tree_to_dict(tree), json_file, indent=4)
            print("✅ JSON file saved!")

        # Evaluate on training data
        train_predictions = predict(tree, X_train)
        train_acc = np.mean(train_predictions == y_train)

        # Evaluate on test data
        test_predictions = predict(tree, X_test)
        test_acc = np.mean(test_predictions == y_test)

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        print(f"📊 Trial {trial + 1}: Training Accuracy = {train_acc:.4f}, Testing Accuracy = {test_acc:.4f}")

    print("\n🎉 All experiments completed!")
    return train_accuracies, test_accuracies

# Load dataset
print("📂 Loading dataset...")
df = pd.read_csv("car.csv")
print("✅ Dataset successfully loaded!")

# Preprocess data
X = df.iloc[:, :-1]  # First 6 columns as features
y = df.iloc[:, -1]   # Last column as labels
print("🔍 Data preprocessing completed!")

# Train and evaluate the model (100 trials)
train_acc, test_acc = evaluate(X, y, test_size=0.2, num_trials=100)

# Compute mean and standard deviation
train_mean, train_std = np.mean(train_acc), np.std(train_acc)
test_mean, test_std = np.mean(test_acc), np.std(test_acc)

print(f"\n📢 Final Results:")
print(f"✅ Training Accuracy Mean: {train_mean:.4f} (Std: {train_std:.4f})")
print(f"✅ Testing Accuracy Mean: {test_mean:.4f} (Std: {test_std:.4f})")

# Plot histograms for accuracy distributions
plt.figure(figsize=(12, 5))

# Training accuracy histogram
plt.subplot(1, 2, 1)
plt.hist(train_acc, bins=10, color='blue', edgecolor='black', alpha=0.7, density=True)
plt.xlabel("Accuracy")
plt.ylabel("Frequency")
plt.title(f"Training Accuracy Distribution\nMean={train_mean:.4f}, Std={train_std:.4f}")

# Testing accuracy histogram
plt.subplot(1, 2, 2)
plt.hist(test_acc, bins=10, color='red', edgecolor='black', alpha=0.7, density=True)
plt.xlabel("Accuracy")
plt.ylabel("Frequency")
plt.title(f"Testing Accuracy Distribution\nMean={test_mean:.4f}, Std={test_std:.4f}")

plt.tight_layout()
plt.savefig('accuracy_histogram.png')
