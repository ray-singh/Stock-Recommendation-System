import math
from typing import TypeVar, Generator, List, Tuple, Optional
from collections import deque
import json
from queue import SimpleQueue
import heapq

T = TypeVar("T")  # represents generic type
# represents a Node object (forward-declare to use in Node __init__)
Node = TypeVar("Node")
# represents a custom type used in application
AVLWrappedDictionary = TypeVar("AVLWrappedDictionary")


class Node:
    """
    Implementation of an BST and AVL tree node.
    Do not modify.
    """
    __slots__ = ["value", "parent", "left", "right", "height"]

    def __init__(self, value: T, parent: Node = None,
                 left: Node = None, right: Node = None) -> None:
        """
        Construct an AVL tree node.

        :param value: value held by the node object
        :param parent: ref to parent node of which this node is a child
        :param left: ref to left child node of this node
        :param right: ref to right child node of this node
        """
        self.value = value
        self.parent, self.left, self.right = parent, left, right
        self.height = 0

    def __repr__(self) -> str:
        """
        Represent the AVL tree node as a string.

        :return: string representation of the node.
        """
        return f"<{str(self.value)}>"

    def __str__(self) -> str:
        """
        Represent the AVL tree node as a string.

        :return: string representation of the node.
        """
        return repr(self)


####################################################################################################

class BinarySearchTree:
    """
    Implementation of an BSTree.
    Modify only below indicated line.
    """

    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["origin", "size"]

    def __init__(self) -> None:
        """
        Construct an empty BST tree.
        """
        self.origin = None
        self.size = 0

    def __repr__(self) -> str:
        """
        Represent the BSTree as a string.

        :return: string representation of the BST tree
        """
        if self.origin is None:
            return "Empty BST Tree"

        lines = pretty_print_binary_tree(self.origin, 0, False, '-')[0]
        return "\n" + "\n".join((line.rstrip() for line in lines))

    def __str__(self) -> str:
        """
        Represent the BSTree as a string.

        :return: string representation of the BSTree
        """
        return repr(self)

    def height(self, root: Node) -> int:
        """
        Calculates and returns the height of a subtree in the BSTree

        Parameters:
            root (Node): The root of the subtree whose height is to be calculated.

        Returns:
            int: height of subtree
        """
        if root is None:
            return -1
        return 1+max(self.height(root.left), self.height(root.right))

    def insert(self, root: Node, val: T) -> None:
        """
        Inserts a node with the value val into the subtree rooted at root

        Parameters:
            root (Node): The root of the subtree into which the node is to be inserted
            val (T): The value to be inserted

        Returns:
            None
        """
        if root is None:
            #tree is empty, create a new node and make it the root
            self.origin = Node(val)
            self.size += 1
            return self.origin

        elif val < root.value:
            #value less than current node, go left
            if root.left is None:
                root.left = Node(val, parent=root)
                self.size+=1
            else:
                self.insert(root.left, val)

        elif val > root.value:
            #value greater than current node, go right
            if root.right is None:
                root.right = Node(val, parent=root)
                self.size+= 1
            else:
                self.insert(root.right, val)

        #update height of the current node
        root.height = max(self.height(root.left), self.height(root.right)) + 1

    def remove(self, root: Node, val: T) -> Optional[Node]:
        """
        Removes the node with the value val from the subtree rooted at root.

        Parameters:
            root (Node): The root of the subtree from which to delete value
            val (T): The value to be deleted from the subtree rooted at root

        Returns:
            Node: root of the new subtree after the removal
        """
        if root is None:
            return root

        if val < root.value:
            root.left = self.remove(root.left, val)
        elif val > root.value:
            root.right = self.remove(root.right, val)
        else:
            # Node to be removed has no children or only one child
            if root.left is None:
                temp = root.right
                if temp:
                    temp.parent = root.parent
                root = None
                self.size -= 1
                if self.size == 0:
                    self.origin = None  # Update origin if the tree becomes empty
                return temp
            elif root.right is None:
                temp = root.left
                if temp:
                    temp.parent = root.parent
                root = None
                self.size -= 1
                if self.size == 0:
                    self.origin = None  # Update origin if the tree becomes empty
                return temp
            else:
                # Node to be removed has two children
                # Find max value in left subtree (predecessor)
                temp = self.max_value(root.left)
                # Swap values between root and predecessor
                root.value = temp.value
                # Recursively remove the predecessor node
                root.left = self.remove(root.left, temp.value)

        if root:
            # Update height of the current node
            root.height = max(self.height(root.left), self.height(root.right)) + 1

        return root

    def max_value(self, node: Node) -> Node:
        """
        Find the node with the maximum value in the tree rooted at the given node.
        Helper function for removing a node

        Parameters:
            node (Node): The root of the subtree whose maximum value is to be found.

        Returns:
            Node: the node with the maximum value in the tree rooted at the given node
        
        """
        current = node
        while current.right is not None:
            current = current.right
        return current

    def search(self, root: Node, val: T) -> Optional[Node]:
        """
        Searches for and returns the Node containing the value val in the subtree rooted at root.

        Parameters:
            root (Node): The root of the subtree being searched
            val (T): The value to be searched for

        Returns:
            Node: node if it exists, otherwise prospective parent.
        """
        if root is None:
            return None

        if val == root.value:
            return root

        elif val < root.value:
            if root.left is None:
                return root
            else:
                return self.search(root.left, val)

        else:
            if root.right is None:
                return root
            else:
                return self.search(root.right, val)


class AVLTree:
    """
    Implementation of an AVL tree.
    Modify only below indicated line.
    """

    __slots__ = ["origin", "size"]

    def __init__(self) -> None:
        """
        Construct an empty AVL tree.
        """
        self.origin = None
        self.size = 0

    def __repr__(self) -> str:
        """
        Represent the AVL tree as a string.

        :return: string representation of the AVL tree
        """
        if self.origin is None:
            return "Empty AVL Tree"

        return super(AVLTree, self).__repr__()

    def __str__(self) -> str:
        """
        Represent the AVLTree as a string.

        :return: string representation of the BSTree
        """
        return repr(self)

    def height(self, root: Node) -> int:
        """
        Calculates the height of a subtree in the AVL tree, handling cases where root might be None.

        Parameters:
            root (Node): The root of the subtree whose height is to be found.

        Returns:
            int: height of the AVL tree
        """
        if root is None:
            return -1
        return 1 + max(self.height(root.left), self.height(root.right))


    def left_rotate(self, root: Node) -> Optional[Node]:
        """
        This method performs a left rotation on the subtree rooted at root,
        returning the new root of the subtree after the rotation.

        Parameters:
            root (Node): The root of the subtree where the rotation is being performed

        Returns:
            Node: root of the new subtree post-rotation.
        """
        if root is None or root.right is None:
            return root

        new_root = root.right
        root.right = new_root.left
        if new_root.left:
            new_root.left.parent = root

        new_root.parent = root.parent
        if root.parent is None:
            self.origin = new_root
        elif root is root.parent.left:
            root.parent.left = new_root
        else:
            root.parent.right = new_root

        new_root.left = root
        root.parent = new_root

        # Update heights
        root.height = 1 + max(self.height(root.left), self.height(root.right))
        new_root.height = 1 + max(self.height(new_root.left), self.height(new_root.right))
        return new_root

    def right_rotate(self, root: Node) -> Optional[Node]:
        """
        This method performs a right rotation on the subtree rooted at root,
        returning the new root of the subtree after the rotation.

        Parameters:
            root (Node): root of the subtree where the rotation is being performed

        Returns:
            Node: The root of the new subtree post-rotation.
        """
        if root is None or root.left is None:
            return root

        new_root = root.left
        root.left = new_root.right
        if new_root.right:
            new_root.right.parent = root

        new_root.parent = root.parent
        if root.parent is None:
            self.origin = new_root
        elif root is root.parent.left:
            root.parent.left = new_root
        else:
            root.parent.right = new_root

        new_root.right = root
        root.parent = new_root

        # Update heights
        root.height = 1 + max(self.height(root.left), self.height(root.right))
        new_root.height = 1 + max(self.height(new_root.left), self.height(new_root.right))
        return new_root

    def balance_factor(self, root: Node) -> int:
        """
        Computes the balance factor of the subtree rooted at root.

        Parameters:
            root (Node):  The root node of the subtree on which to compute the balance factor.

        Returns:
            int: height of left subtree subtracted by height of right subtree
        """
        if root is None:
            return 0

        h_L = root.left.height if root.left else -1
        h_R = root.right.height if root.right else -1

        return h_L-h_R

    def rebalance(self, root: Node) -> Optional[Node]:
        """
        Rebalances the subtree rooted at root if it is unbalanced, and returns
        the root of the resulting subtree post-rebalancing. Considers all four posssible
        imbalance cases.

        Parameters:
            root (Node): The root node of the subtree to rebalance.

        Returns:
            Node: The root of the new, potentially rebalanced subtree.
        """
        if root is None:
            return None

            # Calculate the balance factor
        balance = self.balance_factor(root)

        # Left heavy
        if balance >= 2:
            if self.height(root.left.left) >= self.height(root.left.right):
                # Left-left case
                return self.right_rotate(root)
            else:
                # Left-right case
                root.left = self.left_rotate(root.left)
                return self.right_rotate(root)
        # Right heavy
        elif balance <= -2:
            if self.height(root.right.right) >= self.height(root.right.left):
                # Right-right case
                return self.left_rotate(root)
            else:
                # Right-left case
                root.right = self.right_rotate(root.right)
                return self.left_rotate(root)

        return root

    def insert(self, root: Node, val: T) -> Optional[Node]:
        """
        Inserts a new node with value val into the subtree rooted at root,
        balancing the subtree as necessary, and returns the root of the resulting subtree.

        Parameters:
            root (Node): Root of the subtree to be inserted into
            val (T): Value to insert

        Returns:
            Node: The root of the new, balanced subtree.
        """
        if root is None:
            # If the root is None, create a new node with the given value
            new_node = Node(val)
            self.size += 1
            self.origin = new_node if self.origin is None else self.origin
            return new_node

            # If the value is less than the current node's value, go to the left subtree
        if val == root.value:
            return root
        elif val < root.value:
            root.left = self.insert(root.left, val)
            root.left.parent = root
            # If the value is greater than the current node's value, go to the right subtree
        elif val > root.value:
            root.right = self.insert(root.right, val)
            root.right.parent = root
        else:
            # If the value already exists, do nothing
            return root

            # Update the height of the current node
        root.height = 1 + max(self.height(root.left), self.height(root.right))

        # Rebalance the tree if necessary
        return self.rebalance(root)


    def remove(self, root: Node, val: T) -> Optional[Node]:
        """
        This function removes the node with value val from the subtree
        rooted at root, balances the subtree as necessary, and returns
        the root of the resulting subtree.

        Parameters:
            root (Node): Root of the subtree being removed from
            val (T): Value being removed

        Returns:
            Node: The root of the new, balanced subtree.
        """
        if root is None:
            return None

        if val < root.value:
            root.left = self.remove(root.left, val)
        elif val > root.value:
            root.right = self.remove(root.right, val)
        else:
            if root.left is None:
                self.size -= 1
                # Update origin if the removed node is the root
                if root == self.origin:
                    self.origin = root.right
                return root.right
            elif root.right is None:
                self.size -= 1
                # Update origin if the removed node is the root
                if root == self.origin:
                    self.origin = root.left
                return root.left
            else:
                # Node has two children
                # Find max value node in the left subtree
                predecessor = self.max(root.left)
                # Swap the values of the node to be removed and its predecessor
                root.value, predecessor.value = predecessor.value, root.value
                # Remove the predecessor from the left subtree
                root.left = self.remove(root.left, predecessor.value)

            # Update the height of the current node
        root.height = 1 + max(self.height(root.left), self.height(root.right))
        # Rebalance the tree if necessary
        return self.rebalance(root)

    def min(self, root: Node) -> Optional[Node]:
        """
        Finds the node with the minimum value in the AVL tree recursively.

        Parameters:
            root (Node): The root of the subtree within which to search for the minimum value.

        Returns:
            Node or None: The node with the maximum value if the tree is not empty, otherwise None.
        """
        if root is None:
            return None

        if root.left is None:
            return root
        else:
            return self.min(root.left)


    def max(self, root: Node) -> Optional[Node]:
        """
        Finds the node with the maximum value in the AVL tree recursively.

        Parameters:
            root (Node): The root of the subtree within which to search for the minimum value.

        Returns:
            Node or None: The node with the maximum value if the tree is not empty, otherwise None.
        """
        if root is None:
            return None

        if root.right is None:
            return root
        else:
            return self.max(root.right)

    def search(self, root: Node, val: T) -> Optional[Node]:
        """
        Searches for the Node with the value val within the subtree rooted at root.

        Parameters:
            root (Node): The root of the subtree within which to search for the value
            val (T): The value to be searched for

        Returns:
            Node: Node object containing val if it exists within the subtree, otherwise prospective parent.
        """
        if root is None or root.value == val:
            return root

        if val < root.value:
            # If the value is less than the current node's value, search in the left subtree
            if root.left is None:
                return root  # Return the current node if the value would be inserted here
            else:
                return self.search(root.left, val)
        else:
            # If the value is greater than the current node's value, search in the right subtree
            if root.right is None:
                return root  # Return the current node if the value would be inserted here
            else:
                return self.search(root.right, val)

    def inorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Performs an inorder traversal (left, current, right) of the subtree rooted at root.

        Parameters:
            root (Node): The root node of the current subtree being traversed.

        Returns:
            Generator[Node, None, None]: A generator yielding the nodes of the subtree in inorder
        """
        if root is None:
            return

        yield from self.inorder(root.left)
        yield root
        yield from self.inorder(root.right)

    def __iter__(self) -> Generator[Node, None, None]:
        """
        Makes the AVL tree class iterable

        Returns:
            Generator[Node, None, None]: A generator yielding the nodes of the tree in inorder.
        """
        return self.inorder(self.origin)

    def preorder(self, root: Node) -> Generator[Node, None, None]:
        """
        This function performs a preorder traversal (current, left, right) of the subtree rooted at root

        Parameters:
            root (Node): The root node of the current subtree being traversed.

        Returns:
            Generator[Node, None, None]: A generator yielding the nodes of the subtree in preorder.
        """
        if root is None:
            return

        yield root  # Yield the current node

        # Traverse left subtree
        yield from self.preorder(root.left)

        # Traverse right subtree
        yield from self.preorder(root.right)

    def postorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Performs a postorder traversal (left, right, current) of the subtree rooted at root.

        Parameters:
            root (Node): The root node of the current subtree being traversed

        Returns:
            Generator[Node, None, None]: A generator yielding the nodes of the subtree in postorder
        """
        if root is None:
            return

        # Traverse left subtree
        yield from self.postorder(root.left)

        # Traverse right subtree
        yield from self.postorder(root.right)

        yield root  # Yield the current node

    def levelorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Performs a level-order (breadth-first) traversal of the subtree rooted at root.

        Parameters:
            root (Node): The root node of the current subtree being traversed

        Returns:
            Generator[Node, None, None]: A generator yielding the nodes of the subtree in level-order.
        """
        if root is None:
            return

        queue = SimpleQueue()
        queue.put(root)

        while not queue.empty():
            node = queue.get()
            yield node

            if node.left is not None:
                queue.put(node.left)
            if node.right is not None:
                queue.put(node.right)


####################################################################################################

class User:
    """
    Class representing a user of the stock marker.
    Note: A user can be both a buyer and seller.
    """

    def __init__(self, name, pe_ratio_threshold, div_yield_threshold):
        self.name = name
        self.pe_ratio_threshold = pe_ratio_threshold
        self.div_yield_threshold = div_yield_threshold


####################################################################################################

class Stock:
    __slots__ = ['ticker', 'name', 'price', 'pe', 'mkt_cap', 'div_yield']
    TOLERANCE = 0.001

    def __init__(self, ticker, name, price, pe, mkt_cap, div_yield):
        """
        Initialize a stock.

        :param name: Name of the stock.
        :param price: Selling price of stock.
        :param pe: Price to earnings ratio of the stock.
        :param mkt_cap: Market capacity.
        :param div_yield: Dividend yield for the stock.
        """
        self.ticker = ticker
        self.name = name
        self.price = price
        self.pe = pe
        self.mkt_cap = mkt_cap
        self.div_yield = div_yield

    def __repr__(self):
        """
        Return string representation of the stock.

        :return: String representation of the stock.
        """
        return f"{self.ticker}: PE: {self.pe}"

    def __str__(self):
        """
        Return string representation of the stock.

        :return: String representation of the stock.
        """
        return repr(self)

    def __lt__(self, other):
        """
        Check if the stock is less than the other stock.

        :param other: The other stock to compare to.
        :return: True if the stock is less than the other stock, False otherwise.
        """
        return self.pe < other.pe

    def __eq__(self, other):
        """
        Check if the stock is equal to the other stock.

        :param other: The other stock to compare to.
        :return: True if the stock is equal to the other stock, False otherwise.
        """
        return abs(self.pe - other.pe) < self.TOLERANCE


def make_stock_from_dictionary(stock_dictionary: dict[str: str]) -> Stock:
    """
    Builds an AVL tree with the given stock dictionary.

    :param stock_dictionary: Dictionary of stocks to be inserted into the AVL tree.
    :return: A stock in a Stock object.
    """
    stock = Stock(stock_dictionary['ticker'], stock_dictionary['name'], stock_dictionary['price'], \
                  stock_dictionary['pe_ratio'], stock_dictionary['market_cap'], stock_dictionary['div_yield'])
    return stock

def build_tree_with_stocks(stocks_list: List[dict[str: str]]) -> AVLTree:
    """
    Builds an AVL tree with the given list of stocks.

    :param stocks_list: List of stocks to be inserted into the AVL tree.
    :return: AVL tree with the given stocks.
    """
    avl = AVLTree()
    for stock in stocks_list:
        stock = make_stock_from_dictionary(stock)
        avl.insert(avl.origin, stock)
    return avl

def recommend_stock(stock_tree: AVLTree, user: User, action: str) -> Optional[Stock]:
    '''
    Recommends which stock to buy or sell, depending on user's thresholds.

    Parameters:
        stock_tree (AVL Tree): AVL tree containing stock nodes
        user (User): A user object representing the investor's preferences.
        action (str): A string indicating the desired action, either 'buy' or 'sell'.

    Returns:
        Stock or None: a Stock object representing the recommended stock that best fits
        the user's criteria, None if there's no node that meets criteria.
    '''
    best_stock = None
    best_score = float('-inf') if action == 'buy' else float('inf')

    for node in stock_tree.inorder(stock_tree.origin):
        stock = node.value
        score = stock.div_yield / stock.pe

        if stock.pe < user.pe_ratio_threshold and stock.div_yield > user.div_yield_threshold:
            if score > best_score:
                best_score = score
                best_stock = stock

        elif action == 'sell':
            if stock.pe > user.pe_ratio_threshold or stock.div_yield < user.div_yield_threshold:
                if score < best_score:
                    best_score = score
                    best_stock = stock

    return best_stock

def min_stock(tree: AVLTree):
    '''
    Helper function for prune(). Finds the least valuable stock in the
    tree.

    Parameters:
        tree (AVLTree): The tree within which to search for the minimum value.

    Returns:
        Node: The least valuable stock in the tree.
    '''
    curr = tree.origin
    while curr.left is not None:
        curr = curr.left
    return curr

def prune(stock_tree: AVLTree, threshold: float = 0.05) -> None:
    """
    This function removes subtrees of the given Stock AVL Tree where all
    pe values are less than threshold.

    Parameters:
        stock_tree (AVL_Tree): The AVL Tree to be pruned
        threshold (float): Any subtree with all pe values less than this gets removed.

    Returns:
        None
    """
    min = min_stock(stock_tree)
    while (min.value.pe < threshold):
        if (min == stock_tree.origin):
            if stock_tree.origin.right is not None:
                new_root = stock_tree.origin.right
                new_root.parent = None

            elif stock_tree.origin.left is not None:
                new_root = stock_tree.origin.left
                new_root.origin,parent = None

        stock_tree.remove(stock_tree.origin, min.value)

        if stock_tree.origin is None:
            if new_root.value.pe < threshold:
                return None
            stock_tree.origin = new_root
            stock_tree.size = 1
            return None

        min = min_stock(stock_tree)

    return None


def pretty_print_binary_tree(root: Node, curr_index: int, include_index: bool = False,
                             delimiter: str = "-", ) -> \
        Tuple[List[str], int, int, int]:
    """
    Taken from: https://github.com/joowani/binarytree

    Recursively walk down the binary tree and build a pretty-print string.
    In each recursive call, a "box" of characters visually representing the
    current (sub)tree is constructed line by line. Each line is padded with
    whitespaces to ensure all lines in the box have the same length. Then the
    box, its width, and start-end positions of its root node value repr string
    (required for drawing branches) are sent up to the parent call. The parent
    call then combines its left and right sub-boxes to build a larger box etc.
    :param root: Root node of the binary tree.
    :type root: binarytree.Node | None
    :param curr_index: Level-order_ index of the current node (root node is 0).
    :type curr_index: int
    :param include_index: If set to True, include the level-order_ node indexes using
        the following format: ``{index}{delimiter}{value}`` (default: False).
    :type include_index: bool
    :param delimiter: Delimiter character between the node index and the node
        value (default: '-').
    :type delimiter:
    :return: Box of characters visually representing the current subtree, width
        of the box, and start-end positions of the repr string of the new root
        node value.
    :rtype: ([str], int, int, int)
    .. _Level-order:
        https://en.wikipedia.org/wiki/Tree_traversal#Breadth-first_search
    """
    if root is None:
        return [], 0, 0, 0

    line1 = []
    line2 = []
    if include_index:
        node_repr = "{}{}{}".format(curr_index, delimiter, root.value)
    else:
        if type(root) == HuffmanNode:
            node_repr = repr(root)
        elif type(root.value) == AVLWrappedDictionary:
            node_repr = f'{root.value},h={root.height},' \
                        f'⬆{str(root.parent.value.key) if root.parent else "None"}'
        else:
            node_repr = f'{root.value},h={root.height},' \
                        f'⬆{str(root.parent.value) if root.parent else "None"}'

    new_root_width = gap_size = len(node_repr)

    # Get the left and right sub-boxes, their widths, and root repr positions
    l_box, l_box_width, l_root_start, l_root_end = pretty_print_binary_tree(
        root.left, 2 * curr_index + 1, include_index, delimiter
    )
    r_box, r_box_width, r_root_start, r_root_end = pretty_print_binary_tree(
        root.right, 2 * curr_index + 2, include_index, delimiter
    )

    # Draw the branch connecting the current root node to the left sub-box
    # Pad the line with whitespaces where necessary
    if l_box_width > 0:
        l_root = (l_root_start + l_root_end) // 2 + 1
        line1.append(" " * (l_root + 1))
        line1.append("_" * (l_box_width - l_root))
        line2.append(" " * l_root + "/")
        line2.append(" " * (l_box_width - l_root))
        new_root_start = l_box_width + 1
        gap_size += 1
    else:
        new_root_start = 0

    # Draw the representation of the current root node
    line1.append(node_repr)
    line2.append(" " * new_root_width)

    # Draw the branch connecting the current root node to the right sub-box
    # Pad the line with whitespaces where necessary
    if r_box_width > 0:
        r_root = (r_root_start + r_root_end) // 2
        line1.append("_" * r_root)
        line1.append(" " * (r_box_width - r_root + 1))
        line2.append(" " * r_root + "\\")
        line2.append(" " * (r_box_width - r_root))
        gap_size += 1
    new_root_end = new_root_start + new_root_width - 1

    # Combine the left and right sub-boxes with the branches drawn above
    gap = " " * gap_size
    new_box = ["".join(line1), "".join(line2)]
    for i in range(max(len(l_box), len(r_box))):
        l_line = l_box[i] if i < len(l_box) else " " * l_box_width
        r_line = r_box[i] if i < len(r_box) else " " * r_box_width
        new_box.append(l_line + gap + r_line)

    # Return the new box, its width and its root repr positions
    return new_box, len(new_box[0]), new_root_start, new_root_end


if __name__ == "__main__":
    pass

