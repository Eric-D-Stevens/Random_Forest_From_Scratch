import numpy as np
from tree.node import Node
from tree.tree import Tree

x = np.random.random((100,10))
y = np.atleast_2d(np.random.randint(2, size=100)).T

N = Tree()
N.train(x,y)

x_test = np.random.random((1, 10))
result = N.predict(x=x_test)


print(result)
