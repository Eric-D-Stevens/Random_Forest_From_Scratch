import numpy as np

class Node:

    def __init__(self,
                 data: np.ndarray,
                 labels: np.ndarray,
                 impurity_metric: str = 'gini',
                 depth: int = 0,
                 max_depth: int = 10,
                 min_node_points = 1):
        """
        This class is the basic building block of
        the decision tree. In this implementation,
        and entire decision tree is built upon
        initialization of this class. There is a
        super class "Tree" that is meant to abstract
        the use of this class away from users.

        The class builds instances of itself recursively
        upon initialization setting member attributes to
        instances of its own type (left/right_child). This
        process continues until impurity of a leaf node
        reaches zero or until some regularization threshold,
        such as the depth of a node equaling 'max_depth'
        results in growth termination.


        :param data: Numpy array of training instances
            where rows are instances and columns are
            features for that instance.
        :param labels: Numpy array that has shape number
            of instance of in data for rows and 1 column.
            This is tricky because the user must ensure
            that the input uses np.atleast_2d and transpose
            it before it can be handed into this parameter
        :param impurity_metric: Defaults to 'gini' which is
            all that is currently available. Future plans to
            add entropy support.
        :param depth: The current depth of the node. This
            parameter should be ignored by the user. It aids
            in 'max_depth' regularization and is automatically
            updated as the tree is grown.
        :param max_depth: The tree will stop growing when it
            reaches a depth of 'max_depth'.
        :param min_node_points: Another regularization parameter
            that prevents a leaf from being created if there are
            less than 'min_node_points' instances in that node.
        """

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

        # if a node is 100% pure, no need to spawn
        if self.impurity == 0.0:
            return

        # if a node is not at the max depth, spawn children
        if self.depth < self.max_depth:
            self.split_dim, self.split_threshold, self.gain = self.spawn_children()

    def calc_gini(self) -> float:
        """
        Calculates Gini impurity of the current node.
        :return: Gini impurity
        """
        return 1. - np.sum(np.square(self.class_counts/self.n))

    def spawn_children(self):
        """
        This function, and its helper functions, do the
        heavy lifting of the decision tree learning algorithm.
        First a call is made to the member function 'self.find_split()"
        which returns the dimension and the threshold on that dimension
        that partitions the data in a way that results in the smallest
        Gini impurity.

        The function then partitions the data into two sets based on the
        dimension threshold, and recursively calls Node class creations,
        setting these new objects to its left and right children.

        Regularization occurs in this function when the size of the
        data in either resulting split is less than the regularization
        parameter 'min_node_points'. If this is the case, the spawning
        will be abandon and the calling node will be a leaf.

        :return: Tuple serving no current purpose. All aspects of tree
            growth are managed within the function itself.
        """

        # based on the nodes data, get the best dimension to split on,
        # the threshold that on that dimension that will reduce impurity
        # the most, and the resulting impurity (split_cost)
        split_dimension, split_threshold, split_cost = self.find_split()
        if split_threshold == None:
            return None,None,None
        self.split_threshold = split_threshold
        self.split_dim = split_dimension

        # Parse data based on above calculated split criterion
        # [X,y] -> [X_l,y_l],[X_r,y_r]
        left_indices = np.argwhere(self.data[:, split_dimension] <= split_threshold)
        left_data = self.data[left_indices[:, 0], :]
        left_labels = self.labels[left_indices[:, 0], 0]
        left_labels = np.atleast_2d(left_labels).T
        right_indices = np.argwhere(self.data[:, split_dimension] > split_threshold)
        right_data = self.data[right_indices[:, 0], :]
        right_labels = self.labels[right_indices[:,0], 0]
        right_labels = np.atleast_2d(right_labels).T

        # Regularization: implement least points in leaf node
        if self.n > self.min_node_points:
        #if len(left_indices) < self.min_node_points or \
        #        len(right_indices) < self.min_node_points:
        #        return None, None, None

                    # spawn left child
            if left_data.shape[0] > 0:
                self.left_child = Node(data=left_data,
                                       labels=left_labels,
                                       impurity_metric=self.impurity_metric,
                                       depth=self.depth + 1,
                                       max_depth=self.max_depth,
                                       min_node_points=self.min_node_points)
            # spawn right child
            if right_data.shape[0] > 0:
                self.right_child = Node(data=right_data,
                                        labels=right_labels,
                                        impurity_metric=self.impurity_metric,
                                        depth=self.depth + 1,
                                        max_depth=self.max_depth,
                                        min_node_points=self.min_node_points)

        return split_dimension, split_threshold, split_cost

    def find_split(self):
        """
        This This function and its helpers are responsible for finding
        the best dimension and threshold for a dataset to be split on to
        reduce impurity. For efficiency, the data itself remains static
        and arrays of sorted indices are used to access the data.

        The run time of the find_split function is O(m*n*log(n)) where
        m is the dimensionality of the dataset and n is the number of
        training instances. This is far better than the O(m*n^2) performance
        we would achieve if we were to not use a sorting method.

        In this function an array of indices, sorted by the data contained
        in a dimension is passed to the helper function 'single_dim_split',
        which uses the ordering of the index array to access data and
        determine which split reduces the impurity the best for that dimension.

        The results for each dimension are compared and the best is kept.

        :return: tuple( best split dimension,
            threshold to  split on that dimension,
            and the resulting impurity of that split.
        """

        # return parameters init
        best_impurity = 1.
        best_threshold = None
        best_dimension = None

        # get array of the size of the data but with values
        # corresponding to sorted indices by column.
        sorted_indices = np.argsort(self.data, axis=0)

        # for each column of sorted indices get best split
        for dim in range(sorted_indices.shape[1]):
            dim_indices = np.atleast_2d(sorted_indices[:, dim]).T
            cur_impur, cur_thresh = self.single_dim_split(dim, dim_indices)
            if cur_impur < best_impurity:
                best_impurity = cur_impur
                best_dimension = dim
                best_threshold = cur_thresh

        return best_dimension, best_threshold, best_impurity



    def single_dim_split(self, dim: int, indices: np.ndarray):
        """
        This function will find the best purity threshold for
        a single dimension array of data.

        :param dim: The dimension of the data currently being assessed.
        :param indices: numpy array of sorted indices for all dimensions
        :return: best threshold and resulting impurity
        """

        # get the labels as a dict
        left_label_counts = {l:0 for l in self.class_labels}
        right_label_counts = {l:c for l,c in zip(self.class_labels, self.class_counts)}

        # return param init
        best_threshold = None
        best_impurity = 1.

        # function to quickly get impurity of combined two data sets
        def mini_gini(left_dict, right_dict):
            left_values = np.array(list(left_dict.values()))
            g_left = 1. - np.sum(np.square(left_values/np.sum(left_values)))
            right_values = np.array(list(right_dict.values()))
            g_right = 1. - np.sum(np.square(right_values/sum(right_values)))
            total = sum(left_values) + sum(right_values)
            return (sum(left_values)/total)*g_left + (sum(right_values)/total)*g_right

        # iterate through each sorted index updating split membership
        for i in range(1, self.n):
            left_val = self.data[indices[i-1, 0], dim]
            right_val = self.data[indices[i, 0], dim]
            left_label_counts[self.labels[indices[i-1, 0], 0]] += 1
            right_label_counts[self.labels[indices[i-1, 0], 0]] -= 1
            cost = mini_gini(left_label_counts, right_label_counts)

            # if split results in better purity, keep it
            if cost < best_impurity and \
                    self.min_node_points < i < self.n - self.min_node_points:
                best_impurity = cost
                best_threshold = (left_val+right_val)/2

        return best_impurity, best_threshold

