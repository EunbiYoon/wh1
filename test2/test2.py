import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# 데이터 로드 및 전처리
df = pd.read_csv("car.csv")  # 파일 경로 수정 필요

# shuffle data
df = shuffle(df).reset_index()

X = df.iloc[:, :-1]  # 특징 데이터 (6개 속성)
y = df.iloc[:, -1]   # 라벨 데이터 (클래스)

# 범주형 데이터를 숫자로 변환 (LabelEncoder 없이)
for col in X.columns:
    X[col] = pd.factorize(X[col])[0]  # 각 범주를 고유 숫자로 변환
y = pd.factorize(y)[0]  # 클래스 라벨도 숫자로 변환

# 엔트로피 계산
def entropy(y):
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

# 정보 이득 계산
def information_gain(X, y, feature_idx):
    base_entropy = entropy(y)
    values, counts = np.unique(X[:, feature_idx], return_counts=True)
    weighted_entropy = sum(
        (counts[i] / np.sum(counts)) * entropy(y[X[:, feature_idx] == v])
        for i, v in enumerate(values)
    )
    return base_entropy - weighted_entropy

# 트리 노드 생성
class Node:
    def __init__(self, feature=None, value=None, left=None, right=None, label=None):
        self.feature = feature  # 분할 기준 속성
        self.value = value  # 분할 기준 값
        self.left = left  # 왼쪽 서브트리
        self.right = right  # 오른쪽 서브트리
        self.label = label  # 리프 노드 클래스 값

# 결정 트리 구축
def build_tree(X, y):
    if len(set(y)) == 1:
        return Node(label=y[0])  # 모든 샘플이 동일한 클래스 -> 리프 노드 생성

    if X.shape[1] == 0:
        return Node(label=Counter(y).most_common(1)[0][0])  # 속성이 더 없으면 다수 클래스 반환

    # 최적의 속성 찾기
    gains = [information_gain(X, y, i) for i in range(X.shape[1])]
    best_feature = np.argmax(gains)

    if gains[best_feature] == 0:
        return Node(label=Counter(y).most_common(1)[0][0])  # 정보 이득이 없으면 다수 클래스 반환

    root = Node(feature=best_feature)
    values = np.unique(X[:, best_feature])

    for v in values:
        sub_X = X[X[:, best_feature] == v]
        sub_y = y[X[:, best_feature] == v]

        if len(sub_X) == 0:
            continue

        child = build_tree(sub_X, sub_y)
        if v == values[0]:  # 왼쪽 자식
            root.left = child
        else:  # 오른쪽 자식
            root.right = child

    return root

# 예측 함수
def predict(tree, sample):
    if tree.label is not None:
        return tree.label
    if sample[tree.feature] == 0:
        return predict(tree.left, sample)
    else:
        return predict(tree.right, sample)

# 모델 평가 함수
def evaluate(X_train, y_train, X_test, y_test):
    tree = build_tree(X_train, y_train)
    y_train_pred = np.array([predict(tree, x) for x in X_train])
    y_test_pred = np.array([predict(tree, x) for x in X_test])

    train_accuracy = np.mean(y_train_pred == y_train)
    test_accuracy = np.mean(y_test_pred == y_test)
    
    return train_accuracy, test_accuracy

# 실험 수행 (100회)
train_accuracies, test_accuracies = [], []

for _ in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, shuffle=True)
    train_acc, test_acc = evaluate(X_train, y_train, X_test, y_test)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

# 결과 시각화
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(train_accuracies, bins=10, edgecolor='black', alpha=0.7)
plt.xlabel("Accuracy on Training Data")
plt.ylabel("Frequency")
plt.title("Training Accuracy Distribution")

plt.subplot(1, 2, 2)
plt.hist(test_accuracies, bins=10, edgecolor='black', alpha=0.7)
plt.xlabel("Accuracy on Testing Data")
plt.ylabel("Frequency")
plt.title("Testing Accuracy Distribution")

plt.tight_layout()
plt.savefig('accuracy_histogram.png')

# 평균 및 표준편차 출력
train_mean, train_std = np.mean(train_accuracies), np.std(train_accuracies)
test_mean, test_std = np.mean(test_accuracies), np.std(test_accuracies)

print(f"Training Accuracy: Mean = {train_mean:.4f}, Std = {train_std:.4f}")
print(f"Testing Accuracy: Mean = {test_mean:.4f}, Std = {test_std:.4f}")
