# Project1
INF264 Project1

Install requirements:
$ pip install -r requirements.txt

Folder structure:
- data/
  - wine_dataset.csv
- decision_tree/
  - decisionTree.py
  - treeNode.py
- evaluation.py
- main.py
- preprocessing.py
- README.md
- requirements.txt
- validation.py

NOTE: to reproduce the results from the report, do not change seed parameters. All seeds are 123 by default.

If you want to test run time, uncomment line 37 in validation.py to get build time, and 57-64 in main to get predict time.

------------------------------------------------------------------------------------------------------------
Edit main.py to get the results wanted. Different options for running the code are described below.

Preprocessing:
1. Create X (feature dataset) y (label dataset) by calling load_dataset(). Takes 3 params:
- path: required. Path to the dataset. 
- drop_nan: True by default. This df does in fact not contain NaN values, but generalized for df's that do.
- visualize_df: False by default. Pairplot for visualizing relationships between numerical features, used in report. 

To create a Decisiontree model: 
1. Create an instance of Decisiontree
2. Call learn on the instance. Learn takes several parameters:
- X: required. The dataset without the labels
- y: required. The labels
- impurity measure: either 'entropy' or 'gini'
- prune: False by default. The method returns a pruned tree if true.
- test_size: 0.3 by default. The size of the training data that is seperated the tree will use for pruning instead of learning. 
- seed: 123 by default. Do not change if you want to reproduce results.
3. Call predict to predict the class label of some new data point x.
4. If you want to visualize the tree, call print_tree()


To create a Decisiontree model with optimal hyperparameters: 
1. Run k_fold_cross_validation on the training data
- X: required. The training dataset without the labels
- y: required. The training data labels
- k: 10 by default. The amount of bins the data should be split into. Note that as k increases the runtime increases as well as it fits decisiontree 4 * k times. 
2. Run learn on the instance of decisiontree returned from k_fold_cross_validation, to utilize on all training data.
3. Call predict to predict the class label of some new data point x.
4. If you want to visualize the tree, call print_tree()

------------------------------------------------------------------------------------------------------------