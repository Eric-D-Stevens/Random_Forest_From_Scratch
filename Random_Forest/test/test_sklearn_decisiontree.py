from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from tree.tree import Tree
import numpy as np

iris = load_iris()
X = iris.data[:, 2:]
y = iris.target

print(X.shape)
print(y.shape)

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X,y)

from sklearn.tree import export_graphviz
export_graphviz(
    tree_clf,
    out_file="tree.png",
    class_names=iris.target_names,
    rounded=True,
    filled=True
)

t = Tree(max_depth=2)
Xme = X
Yme = np.atleast_2d(y).T
t.train(Xme,Yme)


print("OH YEAH")
