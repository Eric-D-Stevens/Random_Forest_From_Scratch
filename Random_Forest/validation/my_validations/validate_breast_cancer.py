import numpy as np
from sklearn.datasets import load_breast_cancer
from validation.random_forest_validation import validate_random_forest

# load the iris dataset form sklearn
iris = load_breast_cancer()
X = iris.data[:, :]
print(X.shape)

y = np.atleast_2d(iris.target).T

validate_random_forest(data=X,
                       labels=y,
                       n_trees=100,
                       max_features=20,
                       bootstrap_features=False,
                       max_depth=8,
                       min_leaf_points=5,
                       train_percent=.75)