import numpy as np


class Bagging:

    def __init__(self,
                 data: np.ndarray,
                 labels: np.ndarray,
                 n_bags: int,
                 max_features: int,
                 bootstrap_features: bool = False,
                 seed: int = None):

        # ensure features are sampled with replacement if
        # max features is larger than number of attributes.
        num_features = data.shape[1]
        print("num_fet: {} max_feat {}".format(num_features, max_features))
        assert(num_features >= max_features
               or bootstrap_features)

        # input attributes
        self.data = data
        self.labels = labels
        self.n_bags = n_bags
        self.bag_size = data.shape[0]
        self.max_features = max_features
        self.bootstrap_features = bootstrap_features
        self.seed = None

        # derived attributes
        self.n = data.shape[0]
        self.num_features = data.shape[1]

        # list comprehension of _Bag class objects
        self.bag_list = [_Bag(data_size=self.n,
                              num_features=self.num_features,
                              max_features=self.max_features,
                              bootstrap_features=self.bootstrap_features,)
                         for _ in range(self.n_bags)]

    def get_bag(self, bag_index: int):
        assert(bag_index < len(self.bag_list))
        bag = self.bag_list[bag_index]

        rows = np.atleast_2d(bag.indices).T
        bag_data = self.data[rows, bag.features]
        bag_labels = self.labels[bag.indices]

        return bag_data, bag_labels, bag.features


class _Bag:

    def __init__(self,
                 data_size: int,
                 num_features: int,
                 max_features: int,
                 bootstrap_features: bool):

        # determine how many features will be used
        self.n_features = np.random.randint(low=1, high=max_features+1)

        # get the features
        self.features = np.random.choice(range(num_features),
                                         size=self.n_features,
                                         replace=bootstrap_features)

        # sample index range randomly
        self.indices = np.random.choice(range(data_size),
                                        size=data_size,
                                        replace=True)








