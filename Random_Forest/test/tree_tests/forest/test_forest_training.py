import numpy as np
from sklearn.datasets import load_iris
from forest.forest import Forest

# load the iris dataset form sklearn
iris = load_iris()
X = iris.data[:, :]
y = np.atleast_2d(iris.target).T

f = Forest(data=X,
           labels=y,
           n_trees=100,
           max_features=3,
           bootstrap_features=True,
           max_depth=4,
           min_leaf_points=2)

print('debug')

index = 101
xt = X[index,:]
yt = y[index]


cls, prb = f.predict(xt)
print("actul lable:", yt)
print("predected {} at {}".format(cls,prb))