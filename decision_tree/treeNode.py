class TreeNode:
    def __init__(self, feature: str=None, feature_index: int=None, threshold=None, value: int=None) -> None:
        #, feature=None, root=None, children: list=None, value: int=None, threshold=None, feature_index: int=None, y_indexes=None
        # new_node = TreeNode(feature_index=X.columns.get_loc(best_feature), feature=best_feature, threshold=best_threshold)
        self.feature = feature
        self.feature_index = feature_index
        self.root = None
        self.children = [] # TODO change list=[]
        self.value = value
        self.threshold = threshold
        self.y_indexes = None
    
    def add_child(self, child: 'TreeNode', y_indexes) -> None:
        """Add a new child to the Tree"""
        child.root = self
        child.y_indexes = y_indexes  # store indexes for pruning
        self.children.append(child)

    def remove_children(self) -> None:
        """For pruning. Removes the connection to the children"""
        self.children = []