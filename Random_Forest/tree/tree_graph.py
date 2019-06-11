from tree.tree import Tree
from tree.node import Node
from graphviz import Digraph

def build_graph(tree: Tree):

    node_attr = [('shape', 'box'), ('style', 'filled, rounded'),
                 ('fontname', 'helvetica')] ;

    graph = Digraph('DTree',
                    filename='my_tree.gv',
                    directory='tree_tests\images',
                    format='png',
                    node_attr=node_attr
                    )

    traversal2(tree.head, graph)

    graph.view()


def traversal(node: Node, graph: Digraph):
    print(type(node))
    if node.left_child:
        graph.edge(str(node.impurity), str(node.left_child.impurity))
    if node.right_child:
        graph.edge(str(node.impurity), str(node.right_child.impurity))

    if node.left_child:
        traversal(node.left_child, graph)

    if node.right_child:
        traversal(node.right_child, graph)


def traversal2(node: Node, graph: Digraph):

    cur_str = build_node_string(node)

    if node.left_child:
        lft_str = build_node_string(node.left_child)
        if lft_str == cur_str: return
        graph.edge(cur_str, lft_str)
    if node.right_child:
        rht_str = build_node_string(node.right_child)
        if rht_str == cur_str: return
        graph.edge(cur_str, rht_str)


    if node.left_child:
        traversal2(node.left_child, graph)

    if node.right_child:
        traversal2(node.right_child, graph)

def build_node_string(node):

    impurtiy = node.impurity
    feat_index = node.split_dim
    threshold = node.split_threshold
    samples = node.n
    lables = list(node.class_count_dict.keys())
    counts = list(node.class_count_dict.values())
    predicted_class = max(node.class_count_dict,
                          key=node.class_count_dict.get)
    percent = node.class_count_dict[predicted_class]/sum(node.class_count_dict.values())


    string = ""
    string += r"Node Impurity {0:5.3f}\n".format(impurtiy)
    string += r"Node Samples {}\n".format(samples)
    string += r"Labels {}\n".format(lables)
    string += r"Counts {}\n".format(counts)
    if threshold:
        string += r"Split on dim {0} with threshold{1:7.3f}\n".format(feat_index, threshold)
    string += r"This node predicts class {0:} at {1:7.2f} %".format(predicted_class, percent*100)

    print(string)
    return string


def test_graph():
    graph = Digraph()
    graph.edge('e', 's')
    graph.view()
