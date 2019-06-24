"""
tree

Decision tree class and container for tree objects.
Used for creating decision tree objects, training
objects on data, and making predictions on new data.
"""

# Node is the training algorithm
from tree.node import Node

# numpy used for matrix manipulation
import numpy as np


class Tree:

    def __init__(self,
                 max_depth: int = 10,
                 min_node_points = 1):
        """
        Instantiates the decision tree model object
        :param max_depth: maximum depth decision tree can grow
        :param min_node_points: minimum points a leaf node can contain
        """

        self.head = None
        self.max_depth = max_depth
        self.min_node_points = min_node_points
        self.trained = False

    def train(self, x: np.ndarray, y: np.ndarray):
        """
        Train the Decision Tree on input data
        :param x: Matrix with rows as observations and
            columns as features.
        :param y: A single column matrix with the same number
            of rows as the input parameter x
        """
        assert(x.shape[0] == y.shape[0])
        assert(y.shape[1] == 1)

        self.head = Node(data=x,
                         labels=y,
                         max_depth=self.max_depth,
                         min_node_points=self.min_node_points)

        self.trained = True

    def predict(self, x: np.ndarray):
        return self.get_prediction(self.head, x)

    def get_prediction(self, node, x: np.ndarray):
        #if node.split_threshold == None:
        if not node.right_child or not node.left_child:
            class_precition = max(node.class_count_dict,
                                  key=node.class_count_dict.get)
            percent = node.class_count_dict[class_precition]/node.n
            return class_precition, percent
        else:
            if x[node.split_dim] < node.split_threshold:
                return self.get_prediction(node.left_child, x)
            elif x[node.split_dim] >= node.split_threshold:
                return self.get_prediction(node.right_child, x)


