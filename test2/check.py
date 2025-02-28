import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split

# 엔트로피 계산 함수
def entropy(y):
    class_counts = y.value_counts()
    probabilities = class_counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

# 정보 이득 계산 함수
def information_gain(X, y, feature):
    # 전체 데이터의 엔트로피 계산
    total_entropy = entropy(y)
    
    # 해당 특성(feature)을 기준으로 데이터를 분할
    values = X[feature].unique()
    weighted_entropy = 0
    for value in values:
        subset_y = y[X[feature] == value]
        weighted_entropy += (len(subset_y) / len(y)) * entropy(subset_y)
    
    # 정보 이득 계산
    return total_entropy - weighted_entropy

# 트리 노드를 나타내는 클래스 정의
class Node:
    def __init__(self, feature=None, label=None, children=None):
        self.feature = feature  # 특성 이름
        self.label = label  # 클래스 레이블 (리프 노드일 경우)
        self.children = children if children else {}  # 자식 노드들

# Decision Tree 학습 함수
def build_tree(X, y, features):
    # 만약 모든 데이터가 하나의 클래스에 속한다면 리프 노드를 반환
    if len(y.unique()) == 1:
        return Node(label=y.iloc[0])
    
    # 만약 더 이상 분할할 특성이 없다면 리프 노드를 반환
    if len(features) == 0:
        return Node(label=y.mode()[0])
    
    # 정보 이득이 가장 큰 특성을 찾음
    best_feature = None
    best_info_gain = -np.inf
    for feature in features:
        info_gain = information_gain(X, y, feature)
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = feature
    
    # 해당 특성을 기준으로 데이터를 분할
    tree = Node(feature=best_feature)
    remaining_features = [f for f in features if f != best_feature]
    for value in X[best_feature].unique():
        subset_X = X[X[best_feature] == value].drop(columns=[best_feature])
        subset_y = y[X[best_feature] == value]
        child_node = build_tree(subset_X, subset_y, remaining_features)
        tree.children[value] = child_node
    
    return tree

# 트리 출력 함수
def tree_to_dict(tree):
    if tree.label is not None:
        return {'label': tree.label}
    
    node = {
        'feature': tree.feature,
        'children': {str(value): tree_to_dict(child) for value, child in tree.children.items()}
    }
    return node

# 예측 함수 (트리를 기반으로 예측)
def predict(tree, X):
    predictions = []
    for _, row in X.iterrows():
        node = tree
        while node.label is None:
            # 값이 트리에서 존재하지 않으면 기본적으로 가장 자주 나타나는 클래스를 예측
            if row[node.feature] not in node.children:
                predicted_label = row[node.feature]
                predictions.append(predicted_label)
                break
            node = node.children[row[node.feature]]
        else:
            predictions.append(node.label)
    return np.array(predictions)

# 모델 학습 및 평가 함수
def evaluate(X, y, test_size=0.2, num_trials=100):
    train_accuracies = []
    test_accuracies = []
    
    for trial in range(num_trials):
        # 데이터를 학습 세트와 테스트 세트로 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=None)
        
        # Decision Tree 학습
        tree = build_tree(X_train, y_train, X.columns)
        
        # 학습 데이터 정확도 계산
        train_predictions = predict(tree, X_train)
        train_acc = np.mean(train_predictions == y_train)
        
        # 테스트 데이터 정확도 계산
        test_predictions = predict(tree, X_test)
        test_acc = np.mean(test_predictions == y_test)
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        # 트리 구조를 딕셔너리 형태로 출력
        tree_dict = tree_to_dict(tree)
        
        # JSON 파일로 저장
        with open(f"decision_tree_{trial + 1}.json", 'w') as json_file:
            json.dump(tree_dict, json_file, indent=4)
    
    return train_accuracies, test_accuracies

# 데이터셋 로드 (여기서는 예시로 직접 데이터셋을 만듭니다)
# 실제로는 pd.read_csv()로 데이터를 로드할 수 있습니다.
data = pd.read_csv('car.csv')

df = pd.DataFrame(data)

# 마지막 열을 레이블로 분리
X = df.drop(columns=['class'])
y = df['class']

# 모델 학습 및 평가 실행
train_acc, test_acc = evaluate(X, y, test_size=0.2, num_trials=10)

# 결과 출력
print(f"Train Accuracies: {train_acc}")
print(f"Test Accuracies: {test_acc}")
