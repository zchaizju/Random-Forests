import random
import numpy as np
from scipy.stats import mode
from utils import information_gain, entropy


class BaseTree(object):
    # a base ID3 decision tree learner

    def __init__(self, max_features=np.sqrt, max_depth=10, min_samples_split=2):
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    # use 'fit' and 'predict' in sklearn-style
    def fit(self, x, y):
        self.n_features = x.shape[1]
        self.n_sub_features = int(self.max_features(self.n_features))
        feature_indices = random.sample(range(self.n_features), self.n_sub_features)
        self.trunk = self.build_tree(x, y, feature_indices, 0)

    # build a tree from depth 0
    def build_tree(self, x, y, feature_indices, depth):
        if depth is self.max_depth or len(y) < self.min_samples_split or entropy(y) is 0:
            return mode(y)[0][0]

        feature_index, threshold = find_split(x, y, feature_indices)
        x_true, y_true, x_false, y_false = split(x, y, feature_index, threshold)
        if y_true.shape[0] is 0 or y_false.shape[0] is 0:
            return mode(y)[0][0]

        feature_indices = random.sample(range(self.n_features), self.n_sub_features)
        branch_true = self.build_tree(x_true, y_true, feature_indices, depth+1)
        branch_false = self.build_tree(x_false, y_false, feature_indices, depth+1)
        return Node(feature_index, threshold, branch_true, branch_false)

    def predict(self, x):
        num_samples = x.shape[0]
        y = np.empty(num_samples)
        for j in range(num_samples):
            node = self.trunk
            while isinstance(node, Node):
                if x[j][node.feature_index] <= node.threshold:
                    node = node.branch_true
                else:
                    node = node.branch_false
            y[j] = node
        return y


# traverse all the candidate features and splits in node with O(n*m*log(m)) complexity
def find_split(x, y, feature_indices):
    best_gain = 0
    best_feature_index = 0
    best_threshold = 0
    for feature_index in feature_indices:
        values = sorted(set(x[:, feature_index]))
        for j in range(len(values)-1):
            threshold = (values[j]+values[j+1])/2
            x_true, y_true, x_false, y_false = split(x, y, feature_index, threshold)
            gain = information_gain(y, y_true, y_false)
            if gain > best_gain:
                best_gain = gain
                best_feature_index = feature_index
                best_threshold = threshold
    return best_feature_index, best_threshold


class Node(object):

    def __init__(self, feature_index, threshold, branch_true, branch_false):
        self.feature_index = feature_index
        self.threshold = threshold
        self.branch_true = branch_true
        self.branch_false = branch_false


# split x and y using feature_index and threshold for that feature
def split(x, y, feature_index, threshold):
    x_true = []
    y_true = []
    x_false = []
    y_false = []
    for j in range(len(y)):
        if x[j][feature_index] <= threshold:
            x_true.append(x[j])
            y_true.append(y[j])
        else:
            x_false.append(x[j])
            y_false.append(y[j])
    x_true = np.array(x_true)
    y_true = np.array(y_true)
    x_false = np.array(x_false)
    y_false = np.array(y_false)
    return x_true, y_true, x_false, y_false

