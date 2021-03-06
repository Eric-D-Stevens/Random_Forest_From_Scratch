import numpy as np
import os
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from graphviz import Source

# my files
from tree.tree import Tree
from tree.tree_graph import build_graph

# load the iris dataset form sklearn
iris = load_iris()
X = iris.data[:, 2:]
y = iris.target

# declare shared hyper parameters
max_depth = 100
min_samples = 10


tree_clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples)
tree_clf.fit(X,y)
sk_tree_graph = export_graphviz(tree_clf,
                                out_file=None,
                                class_names=iris.target_names,
                                rounded=True,
                                filled=True)
# build tree graph for sklearn model
folder = os.path.dirname(__file__)
sk_tree_source = Source(sk_tree_graph,
                        filename='skl_treebag.gv',
                        directory=folder+'\images',
                        format='png')
sk_tree_source.render()


# my tree class
t = Tree(max_depth=max_depth, min_node_points=min_samples)
Xme = X
Yme = np.atleast_2d(y).T
t.train(Xme, Yme)

build_graph(t, '.\images')

print(t.predict(np.array([1,7])))
