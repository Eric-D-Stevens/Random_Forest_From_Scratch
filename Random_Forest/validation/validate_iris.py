import numpy as np
from sklearn.datasets import load_iris
from validation.random_forest_validation import validate_random_forest

# load the iris dataset form sklearn
iris = load_iris()
X = iris.data[:, :]
y = np.atleast_2d(iris.target).T

validate_random_forest(data=X,
                       labels=y,
                       n_trees=30,
                       max_features=3,
                       bootstrap_features=True,
                       max_depth=9,
                       min_leaf_points=5,
                       train_percent=.75)

