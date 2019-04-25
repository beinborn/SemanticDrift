# Some functions used from: https://www.geeksforgeeks.org/find-distance-between-two-nodes-of-a-binary-tree/
import pickle
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree, leaves_list
from scipy.spatial.distance import pdist
import numpy as np
global target_langs
target_langs = ["en", "de", "nl", "fr", "es", "bg", "da", "cs", "pl", "pt", "it", "ro", "sk", "sl", "sv", 'lt', 'lv']

global leaves
leaves = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

leave_lang_dict = {}
for i in range(len(leaves)):
    leave_lang_dict[leaves[i]] = target_langs[i]


class Node:
    def __init__(self, data):
        self.data = data
        self.right = None
        self.left = None


def pathToNode(root, path, k):
    if root is None:
        return False

    path.append(root.data)

    if root.data == k:
        return True

    if ((root.left is not None and pathToNode(root.left, path, k)) or
            (root.right is not None and pathToNode(root.right, path, k))):
        return True

    path.pop()
    return False


def distance(root, data1, data2):
    if root:
        path1 = []
        pathToNode(root, path1, data1)

        path2 = []
        pathToNode(root, path2, data2)

        i = 0
        while i < len(path1) and i < len(path2):
            if path1[i] != path2[i]:
                break
            i = i + 1

        return len(path1) + len(path2) - 2 * i
    else:
        return 0


def printTopToBottomPath(curr, parent):
    stk = []
    while curr:
        stk.append(curr)
        curr = parent[curr]

    while len(stk) != 0:
        curr = stk[-1]
        stk.pop(-1)

        # For hard-coded trees
        # if (int(curr.data)<= len(target_langs)):
        #     toprint = target_langs[curr.data -1]
        # else:
        #     toprint = curr.data
        # print(toprint, end=" ")
        print(curr.data, end=" ")

    print()


def printRootToLeaf(root):
    if root is None:
        return

    nodeStack = [root]
    parent = {}
    parent[root] = None
    while len(nodeStack) != 0:

        current = nodeStack[-1]
        nodeStack.pop(-1)

        if (not current.left and
                not current.right):
            printTopToBottomPath(current, parent)

        if current.right:
            parent[current.right] = current
            nodeStack.append(current.right)
        if current.left:
            parent[current.left] = current
            nodeStack.append(current.left)


def pathToNode2(root, path, k):
    if root is None:
        return False

    path.append(root.get_id())
    if root.get_id() == k:
        return True

    if ((not root.left is None and pathToNode2(root.left, path, k)) or
            (not root.right is None and pathToNode2(root.right, path, k))):
        return True

    path.pop()
    return False


def distance2(root, data1, data2):
    if root:
        path1 = []
        pathToNode2(root, path1, data1)

        path2 = []
        pathToNode2(root, path2, data2)

        i = 0
        while i < len(path1) and i < len(path2):

            if path1[i] != path2[i]:
                break
            i = i + 1

        return len(path1) + len(path2) - 2 * i
    else:
        return 0


def get_key(val):
    for key, value in leave_lang_dict.items():
        if val == value:
            return key


def getTreeData(distance_matrix):
    global target_langs

    cluster_results = linkage(pdist(distance_matrix), method="ward")

    testroot, nodelist = to_tree(cluster_results, rd=True)

    ids = testroot.pre_order(lambda x: x.id)

    results = dendrogram(
        cluster_results,
        labels=target_langs,
        count_sort=False,
    )

    order = []
    for i in results['leaves']:
        order.append(target_langs[i])

    return testroot, ids, order



def computeDistanceScore(testroot, rootg, ids, order):
    score = 0
    counter1 = 0
    for i in ids:
        counter2 = 0
        for j in ids:
            if i != j:
                dist1 = distance2(testroot, i, j)
                t1 = get_key(order[counter1])
                t2 = get_key(order[counter2])
                dg = distance(rootg, t1, t2)
                score += (dist1 - dg) ** 2
            counter2 += 1
        counter1 += 1
    return score


def calc_score_for_hardcoded_tree(root, rootg):
    global leaves
    score = 0
    for i in leaves:
        for j in leaves:
            if i != j:
                dist1 = distance(root, i, j)
                dist2 = distance(rootg, i, j)
                score += (dist1 - dist2) ** 2
    return score


def get_distance(distance_matrix, rootg):
    testroot, ids, order = getTreeData(distance_matrix)
    return computeDistanceScore(testroot, rootg, ids, order)


# Hardcoded gold tree
def get_gold_tree():
    rootg = Node(0)
    rootg.left = Node(18)
    rootg.left.left = Node(19)
    rootg.left.right = Node(20)
    # lt lv
    rootg.left.right.left = Node(16)
    rootg.left.right.right = Node(17)

    rootg.left.left.left = Node(21)
    rootg.left.left.right = Node(22)

    # sl bg
    rootg.left.left.left.left = Node(6)
    rootg.left.left.left.right = Node(14)


    rootg.left.left.right.left = Node(23)
    rootg.left.left.right.right = Node(9)
    # cs sk
    rootg.left.left.right.left.left = Node(8)
    rootg.left.left.right.left.right = Node(13)
    # pl
    rootg.left.left.right.right = Node(9)

    # RIGHT
    rootg.right = Node(25)
    rootg.right.left = Node(26)
    # ro
    rootg.right.left.right = Node(12)
    rootg.right.right = Node(27)

    rootg.right.left.left = Node(24)
    # fr
    rootg.right.left.left.right = Node(4)

    rootg.right.left.left.left = Node(28)
    # it
    rootg.right.left.left.left.right = Node(11)

    rootg.right.left.left.left.left = Node(31)
    # pt es
    rootg.right.left.left.left.left.left = Node(10)
    rootg.right.left.left.left.left.right = Node(5)

    rootg.right.right.left = Node(32)
    # en
    rootg.right.right.right = Node(1)

    rootg.right.right.left.left = Node(29)
    # nl de
    rootg.right.right.left.left.left = Node(3)
    rootg.right.right.left.left.right = Node(2)

    rootg.right.right.left.right = Node(30)
    # da sv
    rootg.right.right.left.right.left = Node(7)
    rootg.right.right.left.right.right = Node(15)
    return rootg


# Tree by Rabinovich
def get_rabinovich_tree():
    # Build structure, numbering is not straightforward (sorry...)
    rabi = Node(0)
    rabi.left = Node(18)
    rabi.right = Node(19)
    rabi.left.left = Node(20)
    rabi.left.left.left = Node(21)
    rabi.left.left.left.left = Node(23)
    rabi.left.left.left.left.left = Node(27)

    rabi.left.left.right = Node(24)
    rabi.left.left.right.left = Node(28)

    rabi.left.right = Node(29)
    rabi.right.left = Node(22)
    rabi.right.left.left = Node(25)
    rabi.right.left.left.left = Node(30)
    rabi.right.left.right = Node(31)

    rabi.right.right = Node(26)
    rabi.right.right.left = Node(32)

    # Create leaves
    # sl pl
    rabi.left.left.left.left.left.left = Node(14)
    rabi.left.left.left.left.left.right = Node(9)

    # lv
    rabi.left.left.left.left.right = Node(17)

    # bg
    rabi.left.left.left.right = Node(6)

    # sl  cs
    rabi.left.left.right.left.left = Node(13)
    rabi.left.left.right.left.right = Node(8)

    # pt
    rabi.left.left.right.right = Node(10)

    # lt ro
    rabi.left.right.left = Node(16)
    rabi.left.right.right = Node(12)

    # da sv
    rabi.right.left.left.left.left = Node(7)
    rabi.right.left.left.left.right = Node(15)

    # en
    rabi.right.left.left.right = Node(1)

    # nl de
    rabi.right.left.right.left = Node(3)
    rabi.right.left.right.right = Node(2)

    # es fr
    rabi.right.right.left.left = Node(5)
    rabi.right.right.left.right = Node(4)

    # it
    rabi.right.right.right = Node(11)
    return rabi

def calculate_score_for_random_trees():
    # This code generates 50000 random trees and calculates the average score.
    # This was the procedure recommended by Rabinovich et al., but they used only 1000 iterations.
    # As the variance of the scores is quite high, we increased the number of iterations
    # We obtained the following values and used them below:
    # Mean = 2848
    # Max = 4520
    # Normalized mean = 0.630
    random_scores = []
    random_matrices = []

    rootg = get_gold_tree()
    num_iterations = 50000
    for i in range(num_iterations):
        print(str(i) +" / " + num_iterations)
        distance_matrix = np.random.uniform(low=0.3, high=0.8, size=(17,17) )
        score = get_distance(distance_matrix, rootg)
        random_scores.append(score)
        random_matrices.append(distance_matrix)


    mean = round(np.mean(random_scores))
    print("Average random score: ", mean )
    maxi = max(random_scores)
    print("Highest random score: ", maxi )

    print("Normalized average random score: ", mean/maxi )
    index_min = np.argmax(random_scores)
    matrix = random_matrices[index_min]
    print(matrix.shape)
    print(matrix)
    return mean, maxi, mean / maxi



