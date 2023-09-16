class TreeNode:
    def __init__(self, feature=None, root=None, children: list=[], value: int=None, threshold=None, feature_index: int=None, y_indexes=None) -> None:
        self.feature = feature
        self.root = root
        self.children = children.copy()
        self.value = value
        self.threshold = threshold
        self.feature_index = feature_index
        self.y_indexes = y_indexes
    
    def add_child(self, child: 'TreeNode', y_indexes) -> None:
        """Add a new child to the Tree"""
        child.root = self
        child.y_indexes = y_indexes
        self.children.append(child)

    def remove_children(self) -> None:
        """For pruning. Removes all children from a node so that it becomes a leaf"""
        self.children = []