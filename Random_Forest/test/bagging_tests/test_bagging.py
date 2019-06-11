import numpy as np
import os
from sklearn.datasets import load_iris

# my files
from bagging.bagging import Bagging


# load the iris dataset form sklearn
iris = load_iris()
X = iris.data[:10, :]
y = np.atleast_2d(iris.target[:10]).T

my_bag = Bagging(data=X,
                 labels=y,
                 n_bags=10,
                 max_features=3)







print("debug")


