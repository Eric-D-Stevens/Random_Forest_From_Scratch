import numpy as np
import os
from sklearn.datasets import load_iris

# my files
from bagging.bagging import Bagging


# load the iris dataset form sklearn
iris = load_iris()
sz = iris.data.shape[0]
use = np.random.choice(range(sz), size=10, replace=False)
X = iris.data[use, :]
y = np.atleast_2d(iris.target[use]).T

my_bag = Bagging(data=X,
                 labels=y,
                 n_bags=10,
                 max_features=3)

data, labels, features = my_bag.get_bag(9)






print("debug")


