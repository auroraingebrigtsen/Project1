class TreeNode:
    def __init__(self, feature=None, root=None, children: list=[], value: int=None, threshold=None, feature_index: int=None) -> None:
        self.feature = feature
        self.root = root
        self.children = children.copy()
        self.value = value
        self.threshold = threshold
        self.feature_index = feature_index
    
    def add_child(self, child: 'TreeNode') -> None:
        """Add a new child to the Tree"""
        child.root = self
        self.children.append(child)

    def remove_child():
        """For pruning?"""
        pass 