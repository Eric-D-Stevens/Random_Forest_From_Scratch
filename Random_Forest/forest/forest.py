import numpy as np
from collections import Counter

from tree.tree import Tree
from bagging.bagging import Bagging, _Bag

class Forest:

    def __init__(self,

                 # input
                 data: np.ndarray,
                 labels: np.ndarray,

                 # bagging features
                 n_trees: int,
                 max_features: int,
                 bootstrap_features: bool,

                 # decision tree features
                 max_depth: int,
                 min_leaf_points: int,):

        # input data
        self.data = data
        self.labels = labels

        # bagging features
        self.n_trees = n_trees
        self.max_features = max_features
        self.bootstrap_features = bootstrap_features

        # decision tree features
        self.max_depth = max_depth
        self.min_leaf_points = min_leaf_points

        # declare data ensemble
        self.bag = self._build_bag()

        # plant a forest
        self.tree_bag = []
        for i in range(len(self.bag.bag_list)):
            b_data, b_labels, b_features = self.bag.get_bag(i)
            self.tree_bag.append(_TreeBag(features=b_features,
                                          data=b_data,
                                          labels=b_labels,
                                          max_depth=self.max_depth,
                                          min_leaf_points=self.min_leaf_points))

    def _build_bag(self):
        bag = Bagging(data=self.data,
                      labels=self.labels,
                      n_bags=self.n_trees,
                      max_features=self.max_features,
                      bootstrap_features=self.bootstrap_features)
        return bag


    def predict(self,
                x: np.ndarray,
                vote: str = 'soft'):

        polls = [t.predict(x) for t in self.tree_bag]
        tally = Counter()
        for cls, scr in polls:
            total = 0
            if vote=='soft':
                tally[cls] += scr
                total+= scr
            elif vote=='hard':
                tally[cls] += 1
                total += 1

        predicted_class = max(tally, key=tally.get)
        probability = tally[predicted_class]/total

        return predicted_class, probability

class _TreeBag:

    def __init__(self,
                 features: np.ndarray,
                 data: np.ndarray,
                 labels: np.ndarray,
                 max_depth: int,
                 min_leaf_points: int):

        self.features = features

        self.d_tree = Tree(max_depth=max_depth,
                           min_node_points=min_leaf_points)
        self.d_tree.train(data, labels)

    def predict(self, x):
        x_bag = x[self.features]
        return self.d_tree.predict(x_bag)



