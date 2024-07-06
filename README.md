# Stock-Recommendation-System
This program provides an automated system for managing and recommending stock investments. It allows users to input stock data and their investment preferences, such as desired P/E ratio and dividend yield thresholds. The program then organizes the stock data efficiently and uses it to analyze and suggest stocks that meet the user's criteria for buying or selling. This enhances the user's ability to make informed decisions about their stock investments, saving time and improving accuracy in the selection process.

This program was created as a learning activity for my data structures and algorithm class. It is supported by my implementation of a Binary Search Tree (BST), AVL Tree, and their corresponding insertion, deletion, traversal, etc algorithms. 

## Classes
### Node
This class represents a node in a binary tree (BST or AVL tree). It contains:
- value: The value held by the node.
- parent, left, right: References to the parent, left child, and right child nodes.
- height: The height of the node in the tree.

### BinarySearchTree 
This class implements a basic Binary Search Tree (BST):
- height(root): Calculates the height of the tree/subtree rooted at root.
- insert(root, val): Inserts a value into the BST.
- remove(root, val): Removes a value from the BST.
- max_value(node): Finds the node with the maximum value in the subtree rooted at node.
- search(root, val): Searches for a value in the BST.

### AVLTree
This class extends the functionality of a BST to create an AVL Tree, which is a self-balancing binary search tree:
- height(root): Calculates the height of the tree/subtree rooted at root.
- left_rotate(root): Performs a left rotation on the subtree rooted at root.
- right_rotate(root): Performs a right rotation on the subtree rooted at root.
- balance_factor(root): Calculates the balance factor of the node root.
- rebalance(root): Rebalances the subtree rooted at root if necessary.
- insert(root, val): Inserts a value into the AVL Tree.
- remove(root, val): Removes a value from the AVL Tree.
- min(root), max(root): Find the minimum and maximum values in the subtree rooted at root.
- search(root, val): Searches for a value in the AVL Tree.
- inorder(root), preorder(root), postorder(root), levelorder(root): Generator methods for traversing the tree.

### Stock
This class represents a stock with various attributes:
- name: Name of the stock.
- price: Selling price of stock.
- pe: Price to earnings ratio of the stock.
- mkt_cap: Market capacity.
- div_yield: Dividend yield for the stock.

### User
This class represents a user with preferences for stock recommendations. A user can be a both buyer or seller. Attributes:
- name: User's name.
- pe_ratio_threshold: User's threshold for desired P/E ratio.
- div_yield_threshold: User's threshold for dividend yield.

## Functions
### make_stock_from_dictionary(stock_dictionary)
Builds an AVL tree with the given stock dictionary. Parameter:
- stock_dictionary: Dictionary of stock information used to create a new Stock object and define its attributes.

### build_tree_with_stocks(stocks_list: List[dict[str: str]])
Builds an AVL tree with the given list of stocks, where each node is a Stock object. Parameter:
- tocks_list: List of stocks to be inserted into the AVL tree

### recommend_stock(stock_tree: AVLTree, user: User, action: str)
Recommends a stock for either buying or selling based on a user's specified thresholds and preferences. This function simplifies the decision-making process for users by automatically filtering stocks based on their specific buying or selling preferences. For each stock, it calculates a score that combines its dividend yield and price-to-earnings ratio (PE ratio). If the user wants to buy (action == 'buy'), it looks for stocks with a low PE ratio and high dividend yield. It picks the stock with the best score (highest score). If the user wants to sell (action == 'sell'), it looks for stocks with a high PE ratio or low dividend yield. It picks the stock with the best score (lowest score). It suggests the stock that best matches the user's criteria based on whether they want to buy or sell. If no stock meets the criteria, it doesn't recommend any stock. 

Parameters:
- stock_tree (AVL Tree): AVL tree containing stock nodes.
- user (User): A user object representing the investor's preferences.
- action (str): A string indicating the desired action, either 'buy' or 'sell'.

### prune((stock_tree: AVLTree, threshold: float)
This function removes subtrees of the given Stock AVL Tree where all P/E Ration values are less than threshold. Parameters:
- stock_tree (AVL_Tree): The AVL Tree to be pruned
- threshold (float): Any subtree with all pe values less than this gets removed.

