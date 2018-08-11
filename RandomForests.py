import numpy as np
from scipy.stats import mode
from utils import shuffle_together
from BaseTree import BaseTree


class RandomForests():
    # an ensemble forests learner

    def __init__(self, n_estimators=10, max_features=np.sqrt, max_depth=10, min_samples_split=2, bootstrap=0.75):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.forest = []

    # use 'fit' and 'predict' in sklearn-style
    def fit(self, x, y):
        self.forest = []
        n_samples = len(y)
        n_sub_samples = round(n_samples*self.bootstrap)

        for i in range(self.n_estimators):
            shuffle_together(x, y)
            X_subset = x[:n_sub_samples]
            y_subset = y[:n_sub_samples]

            tree = BaseTree(self.max_features, self.max_depth, self.min_samples_split)
            tree.fit(X_subset, y_subset)
            self.forest.append(tree)
            print('{} / {} tree built.'.format(i+1, self.n_estimators))

    def predict(self, x):
        n_samples = x.shape[0]
        n_trees = len(self.forest)
        predictions = np.empty([n_trees, n_samples])
        for i in range(n_trees):
            predictions[i] = self.forest[i].predict(x)
        return mode(predictions)[0][0]

