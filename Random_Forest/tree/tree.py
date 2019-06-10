import numpy as np
from tree.node import Node

class Tree:

    def __init__(self, max_depth: int = 10):
        self.head = None
        self.max_depth = max_depth
        self.traied = False

    def train(self, x: np.ndarray, y: np.ndarray):
        assert(x.shape[0] == y.shape[0])
        assert(y.shape[1] == 1)

        self.head = Node(data=x,
                         labels=y,
                         max_depth=self.max_depth)


    def predict(self, x: np.ndarray):
        return self.get_prediction(self.head, x)



    def get_prediction(self, node, x: np.ndarray):
        print("XSIZE", x.size)
        if node.split_threshold == None:
            print(node.class_count_dict)
            class_precition = max(node.class_count_dict,
                                  key=node.class_count_dict.get)
            percent = node.class_count_dict[class_precition]/node.n
            return class_precition, percent

        else:
            if x[0, node.split_dim] < node.split_threshold:
                return self.get_prediction(node.left_child, x)
            elif x[0,node.split_dim] >= node.split_threshold:
                return self.get_prediction(node.right_child, x)


