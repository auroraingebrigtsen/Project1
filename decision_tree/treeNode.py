class TreeNode:
    def __init__(self, feature, root=None, children: list=[], value: int=None, ig: float=None) -> None:
        self.feature = feature
        self.root = root
        self.children = children.copy()
        self.value = value
        self.ig = ig
    
    def add_child(self, child: 'TreeNode') -> None:
        """Add a new child to the Tree"""
        child.root = self
        self.children.append(child)

    def remove_child():
        """For pruning?"""
        pass 