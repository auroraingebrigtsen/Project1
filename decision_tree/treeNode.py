class TreeNode:
    def __init__(self, feature=None, root=None, children: list=[], value: int=None, threshold=None, feature_index: int=None, y_indexes=None) -> None:
        self.feature = feature
        self.feature_index = feature_index
        self.root = root
        self.children = children.copy()
        self.value = value
        self.threshold = threshold
        self.y_indexes = y_indexes
    
    def add_child(self, child: 'TreeNode', y_indexes) -> None:
        """Add a new child to the Tree"""
        child.root = self
        child.y_indexes = y_indexes  # store indexes for pruning
        self.children.append(child)

    def remove_children(self) -> None:
        """For pruning. Removes the connection to the children"""
        self.children = []