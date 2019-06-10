from tree.tree import Tree
from tree.node import Node
def traversal(node: Node, func: function):

    left = node.left_child
    right = node.right_child

    func(left)
    func(right)

    return traversal(left, func)
