import numpy as np

class Node:

    def __init__(self,
                 data: np.ndarray,
                 labels: np.ndarray,
                 impurity_metric: str = 'gini',
                 depth: int = 0,
                 max_depth: int = 10,
                 min_node_points = 1):

        # node instance attributes
        self.terminal = False
        self.n = data.shape[0]
        self.data = data
        self.labels = labels
        self.impurity_metric = impurity_metric
        self.depth = depth
        self.max_depth = max_depth
        self.min_node_points = min_node_points

        # spawn dependent
        self.left_child = None
        self.right_child = None
        self.split_dim = None
        self.split_threshold = None
        self.gain = None

        # class breakdown
        self.class_labels, self.class_counts = \
            np.unique(self.labels, return_counts=True)

        self.class_count_dict = {l:c for l,c in zip(self.class_labels, self.class_counts)}

        # used for predict
        self.best_label = max(self.class_count_dict, key=self.class_count_dict.get)
        self.best_percent = self.class_count_dict[self.best_label]/sum(self.class_counts)


        # obtain impurity (possibly refactor to dict based single line)
        if self.impurity_metric == 'gini':
            self.impurity = self.calc_gini()

        #print("IMPURITY:", self.impurity)
        if self.impurity == 0.0:
            return

        if self.depth < self.max_depth:
            self.split_dim, self.split_threshold, self.gain = self.spawn_children()

    def calc_gini(self) -> float:
        return 1. - np.sum(np.square(self.class_counts/self.n))

    def spawn_children(self):

        split_dimension, split_threshold, split_cost = self.find_split()
        self.split_threshold = split_threshold
        self.split_dim = split_dimension

        left_indices = np.argwhere(self.data[:, split_dimension] <= split_threshold)
        left_data = self.data[left_indices[:, 0], :]
        left_labels = self.labels[left_indices[:, 0], 0]
        left_labels = np.atleast_2d(left_labels).T
        right_indices = np.argwhere(self.data[:, split_dimension] > split_threshold)
        right_data = self.data[right_indices[:, 0], :]
        right_labels = self.labels[right_indices[:,0], 0]
        right_labels = np.atleast_2d(right_labels).T

        # implement least points in leaf node
        if len(left_indices) < self.min_node_points or \
                len(right_indices) < self.min_node_points:
                return None, None, None

        if left_data.shape[0] > 0:
            self.left_child = Node(data=left_data,
                                   labels=left_labels,
                                   impurity_metric=self.impurity_metric,
                                   depth=self.depth+1,
                                   max_depth=self.max_depth,
                                   min_node_points=self.min_node_points)
        if right_data.shape[0] > 0:
            self.right_child = Node(data=right_data,
                                labels=right_labels,
                                impurity_metric=self.impurity_metric,
                                depth=self.depth+1,
                                max_depth=self.max_depth,
                                min_node_points=self.min_node_points)

        return split_dimension, split_threshold, split_cost

    def find_split(self):
        best_impurity = 1.
        best_threshold = None
        best_dimension = None

        sorted_indices = np.argsort(self.data, axis=0)
        for dim in range(sorted_indices.shape[1]):
            dim_indices = np.atleast_2d(sorted_indices[:, dim]).T
            cur_impur, cur_thresh = self.single_dim_split(dim, dim_indices)
            if cur_impur < best_impurity:
                best_impurity = cur_impur
                best_dimension = dim
                best_threshold = cur_thresh

        return best_dimension, best_threshold, best_impurity

    def single_dim_split(self, dim: int, indices: np.ndarray):

        left_label_counts = {l:0 for l in self.class_labels}
        right_label_counts = {l:c for l,c in zip(self.class_labels, self.class_counts)}


        best_threshold = None
        best_impurtiy = 1.

        def mini_gini(left_dict, right_dict):
            left_values = np.array(list(left_dict.values()))
            g_left = 1. - np.sum(np.square(left_values/np.sum(left_values)))
            right_values = np.array(list(right_dict.values()))
            g_right = 1. - np.sum(np.square(right_values/sum(right_values)))
            total = sum(left_values) + sum(right_values)
            return (sum(left_values)/total)*g_left + (sum(right_values)/total)*g_right

        for i in range(1, self.n):
            left_val = self.data[indices[i-1, 0], dim]
            right_val = self.data[indices[i, 0], dim]
            left_label_counts[self.labels[indices[i-1, 0], 0]] += 1
            right_label_counts[self.labels[indices[i-1, 0], 0]] -= 1
            cost = mini_gini(left_label_counts, right_label_counts)
            #print("COST, BST_IMPUR",cost, best_impurtiy, i)
            if cost < best_impurtiy:
                best_impurtiy = cost
                best_threshold = (left_val+right_val)/2

        return best_impurtiy, best_threshold

