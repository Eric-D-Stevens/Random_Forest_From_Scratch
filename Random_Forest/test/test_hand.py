from tree.tree import Tree
import numpy as np

xy = np.array([[1, 1, 1],
              [1, 2, 1],
              [4, 6, 1],
              [1, 2, 1],
              [1, 3, 1],
              [5, 3, 1],
              [-1, -1, 0],
              [-2, -4, 0],
              [-11, -5 ,0],
              [-2, -20, 0]])


X = xy[:,:2]
Y = np.atleast_2d(xy[:,2]).T

assert(Y.shape[0] == X.shape[0])

t = Tree()
t.train(X,Y)

p = t.predict(np.atleast_2d(np.array([-2,40])))


print(p)