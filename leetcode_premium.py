"""
##588
Design an in-memory file system to simulate the following functions:
ls: Given a path in string format. If it is a file path, return a list that only contains this file's name. If it is a directory path, return the list of file and directory names in this directory. Your output (file and directory names together) should in lexicographic order.
mkdir: Given a directory path that does not exist, you should make a new directory according to the path. If the middle directories in the path don't exist either, you should create them as well. This function has void return type.
addContentToFile: Given a file path and file content in string format. If the file doesn't exist, you need to create that file containing given content. If the file already exists, you need to append given content to original content. This function has void return type.
readContentFromFile: Given a file path, return its content in string format.
Input:
["FileSystem","ls","mkdir","addContentToFile","ls","readContentFromFile"]
[[],["/"],["/a/b/c"],["/a/b/c/d","hello"],["/"],["/a/b/c/d"]]
Output:
[null,[],null,null,["a"],"hello"]
"""

# Time:  ls: O(l + klogk), l is the path length, k is the number of entries in the last level directory
#        mkdir: O(l)
#        addContentToFile: O(l + c), c is the content size
#        readContentFromFile: O(l + c)
# Space: O(n + s), n is the number of dir/file nodes, s is the total content size.


class trie:
    def __init__(self):
        self.isFile = False
        self.children = {}
        self.content = ""

class fileSystem:
    def __init__(self):
        self.root = trie()

    def ls(self, path):
        cur = self.root
        cur = self.getNode(path)
        if cur.isFile:
            return self._split(path, "/")[-1]
        return cur.children.keys()

    def mkdir(self, path):
        cur = self.putNode(path)
        cur.isFile = False

    def addContentToFile(self, content, path):
        cur = self.putNode(path)
        cur.isFile = True
        cur.content += content

    def readContentFromFile(self, path):
        cur = self.getNode(path)
        return cur.content

    def putNode(self, path):
        cur = self.root
        for s in self._split(path, "/"):
            if s not in cur.children:
                cur.children[s] = trie()
            cur = cur.children[s]
        return cur

    def getNode(self, path):
        cur = self.root
        for s in self._split(path, "/"):
            cur = cur.children[s]
        return cur

    def _split(self, path, delim):
        if path == "/":
            return []
        return path.split(delim)[1:]

# f = fileSystem()
# print(f.ls("/"))
# f.mkdir("/a/b/c")
# f.addContentToFile("hello", "/a/b/c")
# print(f.ls("/"))
# print(f.readContentFromFile("/a/b/c"))

##159 Longest Substring with At Most Two Distinct Characters
# Input: "eceba"
# Output: 3
# Explanation: t is "ece" which its length is 3.

def lenOfLongestSubStringTwoDistinct(s):
    start = 0
    end = 0
    dict1 = {}
    maxLen = 0
    while end < len(s):
        dict1[s[end]] = end
        if len(dict1) > 2:
            minIdex = min(dict1.values())
            del dict1[s[minIdex]]
            start = minIdex + 1
        maxLen = max(maxLen, end - start + 1)
        end += 1
    return maxLen

# s = 'eceba'
# print(lenOfLongestSubStringTwoDistinct(s))

##1214. Two Sum BSTs
# Description
# Given two binary search trees, return True if and only if there is a node in the first tree and a node in the
# second tree whose values sum up to a given integer target.
# Input: root1 = [2,1,4], root2 = [1,0,3], target = 5
# Output: true
# Explanation: 2 and 3 sum up to 5.

##with inorder iterative

def twoSumBSTs(root1, roo2, target):
    stack1 = []
    set1 = set()
    while stack1 or root1:
        while root1:
            stack1.append(root1)
            root1 = root1.left
        root1 = stack1.pop()
        set1.add(target-root1.val)
        root1 = root1.right

    stack2 = []
    while stack2 or root2:
        while root2:
            stack2.append(root2)
            root2 = root2.left
        root2 = stack2.pop()
        if root2.val in set1:
            return True
        root2 = root2.right

    return False

##Time complexity O(n1 +n2)

##1099. Two Sum Less Than K

# Given an array nums of integers and integer k, return the maximum sum such that there exists i < j with
# nums[i] + nums[j] = sum and sum < k. If no i, j exist satisfying this equation, return -1.
# Input: nums = [34,23,1,24,75,33,54,8], k = 60
# Output: 58
# Explanation: We can use 34 and 24 to sum 58 which is less than 60.
#Two pointer approach

def twoSumLessThanK(nums, k):
    nums.sort()
    ans = -1

    left = 0
    right = len(nums) - 1

    while left < right:
        Sum = nums[left] + nums[right]
        if Sum < k:
            ans = max(ans, Sum)
            left += 1
        else:
            right -= 1

    return ans

# nums = [34,23,1,24,75,33,54,8]
# k = 60
# print(twoSumLessThanK(nums, k))

#256. Paint House
# There is a row of n houses, where each house can be painted one of three colors: red, blue, or green.
# The cost of painting each house with a certain color is different. You have to paint all the houses such
# that no two adjacent houses have the same color.
#
# The cost of painting each house with a certain color is represented by a n x 3 cost matrix. For example,
# costs[0][0] is the cost of painting house 0 with the color red; costs[1][2] is the cost of painting house 1
# with color green, and so on... Find the minimum cost to paint all houses.


# Input: costs = [[17,2,17],[16,16,5],[14,3,19]]
# Output: 10
# Explanation: Paint house 0 into blue, paint house 1 into green, paint house 2 into blue.
# Minimum cost: 2 + 5 + 3 = 10.

def paintHouse(costs):
    if len(costs) == 0 or len(costs[0]) == 0:
        return 0
    if len(costs) == 1:
        return min(costs[0])

    for i in range(1, len(costs)):
        costs[i][0] += min(costs[i-1][1], costs[i-1][2])
        costs[i][1] += min(costs[i-1][0], costs[i-1][2])
        costs[i][2] += min(costs[i-1][0], costs[i-1][1])

    return min(costs[-1])

#Time complexity O(n) and space complexity O(1)
# costs = [[17,2,17],[16,16,5],[14,3,19]]
# print(paintHouse(costs))


#276. Paint Fence
# There is a fence with n posts, each post can be painted with one of the k colors.
# You have to paint all the posts such that no more than two adjacent fence posts have the same color.
# Return the total number of ways you can paint the fence.
# Note:
# n and k are non-negative integers.

# Input: n = 3, k = 2
# Output: 6
# Explanation: Take c1 as color 1, c2 as color 2. All possible ways are:
#
#             post1  post2  post3
#  -----      -----  -----  -----
#    1         c1     c1     c2
#    2         c1     c2     c1
#    3         c1     c2     c2
#    4         c2     c1     c1
#    5         c2     c1     c2
#    6         c2     c2     c1


def paintFence(n, k):
    if n == 0:
        return 0
    if n == 1:
        return k
    same = 0
    diff = k
    for i in range(2, n+1):
        total = same + diff
        same = diff
        diff = (total)*(k-1)
    return same + diff

##Time complexity O(n) and space complexity O(1)
# n = 4
# k = 3
# print(paintFence(n, k))


##186. Reverse Words in a String II
# Given an input string , reverse the string word by word.
# Example:
# Input:  ["t","h","e"," ","s","k","y"," ","i","s"," ","b","l","u","e"]
# Output: ["b","l","u","e"," ","i","s"," ","s","k","y"," ","t","h","e"]
# Note:
# A word is defined as a sequence of non-space characters.
# The input string does not contain leading or trailing spaces.
# The words are always separated by a single space.
# Follow up: Could you do it in-place without allocating extra space?

def reverse(s, left, right):
    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1

def reverse_word(s):
    start = 0
    end = 0
    while end < len(s):
        while end < len(s) and s[end] != " ":
            end += 1
        reverse(s, start, end-1)
        start = end + 1
        end += 1

def reverseWordInStr2(s):
    reverse(s, 0, len(s)-1)
    reverse_word(s)
    return s

##Time complexity O(n) and space complexity O(1)
# s = ["t","h","e"," ","s","k","y"," ","i","s"," ","b","l","u","e"]
# print(reverseWordInStr2(s))


#545. Boundary of Binary Tree
# A binary tree boundary is the set of nodes (no duplicates) denoting the union of the left boundary, leaves, and
# right boundary.
#
# The left boundary is the set of nodes on the path from the root to the left-most node. The right boundary is the
# set of nodes on the path from the root to the right-most node.
#
# The left-most node is the leaf node you reach when you always travel to the left subtree if it exists and the
# right subtree if it doesn't. The right-most node is defined in the same way except with left and right exchanged. ' \
#                          'Note that the root may be the left-most and/or right-most node.
#
# Given the root of a binary tree, return the values of its boundary in a counter-clockwise direction starting
# from the root.

def boundaryOfBinaryTree(root):
    def dfs_leftMost(node):
        if not node.left and node.right:
            return
        res.append(node.val)
        if node.left:
            dfs_leftMost(node.left)
        else:
            dfs_leftMost(node.right)

    def dfs_leaves(node):
        if not node.left and not node.right:
            res.append(node.val)
        if node.left:
            dfs_leaves(node.left)
        if node.right:
            dfs_leaves(node.right)

    def dfs_rightMost(node):
        if not node.left and not node.right:
            return
        if node.right:
            dfs_rightMost(node.right)
        else:
            dfs_rightMost(node.left)
        res.append(node.val)

    if root is None:
        return []

    res = [root.val]
    if root.left:
        dfs_leftMost(root.left)
        dfs_leaves(root.left)
    if root.right:
        dfs_leftMost(root.right)
        dfs_rightMost(root.right)

    return res


##1085. Sum of Digits in the Minimum Number
# Given an array A of positive integers, let S be the sum of the digits of the minimal element of A.
# Return 0 if S is odd, otherwise return 1.
# Input: [34,23,1,24,75,33,54,8]
# Output: 0
# Explanation:
# The minimal element is 1, and the sum of those digits is S = 1 which is odd, so the answer is 0.
#
# Example 2:
#
# Input: [99,77,33,66,55]
# Output: 1
# Explanation:
# The minimal element is 33, and the sum of those digits is S = 3 + 3 = 6 which is even, so the answer is 1.
def sumOfDigits(A):
    num = min(A)
    Sum = 0
    while num > 0:
        rem = num % 10
        num = num // 10
        Sum += rem
    if Sum % 2 == 0:
        return 1
    else:
        return 0

# A = [34,23,1,24,75,33,54,8]
# print(sumOfDigits(A))

##293. Flip Game
# You are playing the following Flip Game with your friend: Given a string that contains only these two
# characters: + and -, you and your friend take turns to flip two consecutive "++" into "--". The game ends when
# a person can no longer make a move and therefore the other person will be the winner.
# Write a function to compute all possible states of the string after one valid move.

# Example:
# Input:
# s = "++++"
# Output:
# [
#   "--++",
#   "+--+",
#   "++--"
# ]

def flipGame(s):
    n = len(s)
    s = list(s)
    moves = []

    for i in range(n-1):
        if s[i] == "+" and s[i+1] == "+":
            s[i] = "-"
            s[i+1] = "-"
            moves.append("".join(s))
            ##Recovery
            s[i] = "+"
            s[i+1] = "+"
    return moves
# s = "++++"
# print(flipGame(s))


##294. Flip Game II
# You are playing the following Flip Game with your friend: Given a string that contains only these two characters:
# + and -, you and your friend take turns to flip two consecutive "++" into "--". The game ends when a person can
# no longer make a move and therefore the other person will be the winner.
# Write a function to determine if the starting player can guarantee a win.
# Example:
# Input:
# s = "++++"
# Output: true
# Explanation: The starting player can guarantee a win by flipping the middle
# "++"
#  to become
# "+--+"
# .
# Follow up:
# Derive your algorithm's runtime complexity.

def flipGame2(s):
    memo = {}
    return filpgame2memo(s, memo)


def filpgame2memo(s, memo):
    if s is None or len(s) < 2:
        return False
    if s in memo:
        return memo[s]
    for i in range(len(s)-1):
        if s[i] == "+" and s[i+1] == "+":
            nextState = s[:i] + "--" + s[i+2:]
            if not filpgame2memo(nextState, memo):
                memo[s] = True
                return True
    memo[s] = False
    return False

# s = "++++"
# print(flipGame2(s))


##156. Binary Tree Upside Down
# Given the root of a binary tree, turn the tree upside down and return the new root.
# You can turn a binary tree upside down with the following steps:
# The original left child becomes the new root.
# The original root becomes the new right child.
# The original right child becomes the new left child.

def upSideDownBinaryTree(root):
    if root is None:
        return None
    if root.left is None:
        return root
    new_root = upSideDownBinaryTree(root.left)
    root.left.left = root.right
    root.left.right = root
    root.left = None
    root.right = None
    return new_root


# 1772. Sort Features by Popularity
# You are given a string array features where features[i] is a single word that represents the name of a feature of
# the latest product you are working on. You have made a survey where users have reported which features they like.
# You are given a string array responses, where each responses[i] is a string containing space-separated words.
# The popularity of a feature is the number of responses[i] that contain the feature. You want to sort the features
# in non-increasing order by their popularity. If two features have the same popularity, order them by their
# original index in features. Notice that one response could contain the same feature multiple times; this
# feature is only counted once in its popularity.
# Return the features in sorted order.

# Input: features = ["cooler","lock","touch"],
# responses = ["i like cooler cooler","lock touch cool","locker like touch"]
# Output: ["touch","cooler","lock"]
# Explanation: appearances("cooler") = 1, appearances("lock") = 1, appearances("touch") = 2.
# Since "cooler" and "lock" both had 1 appearance, "cooler" comes first because "cooler" came first in the
# features array.

def byPopularity(features, responses):
    featureDict = {}
    for word in features:
        featureDict[word] = 0


    for sentance in responses:
        visited = set()
        sentance = sentance.split()
        for word in sentance:
            if word not in visited and word in featureDict:
                featureDict[word] += 1
                visited.add(word)

    def compare(str1, str2):
        if featureDict[str1] > featureDict[str2]:
            return -1
        elif featureDict[str1] < featureDict[str2]:
            return 1
        else:
            return 0

    # print(featureDict)
    res = [[word, featureDict[word]] for word in features]
    print(res)
    res = sorted(res, key = lambda x: x[-1], reverse=True)
    result = []
    for i in res:
        result.append(i[0])
    return result

    #return features

# features = ["a","aa","b","c"]
# responses = ["a","a aa","a a a a a","b a"]
# print(byPopularity(features, responses))

##270. Closest Binary Search Tree Value
# Given a non-empty binary search tree and a target value, find the value in the BST that is closest to the target.
# Note:
# Given target value is a floating point.
# You are guaranteed to have only one unique value in the BST that is closest to the target.
# Example:
# Input: root = [4,2,5,1,3], target = 3.714286
#
#     4
#    / \
#   2   5
#  / \
# 1   3
#
# Output: 4

def closestValue(root):
    if root is None:
        return None
    closest = root.val
    while root:
        if abs(target-root.val) < abs(target-closest):
            closest = root.val
        if target < root.val:
            root = root.left
        else:
            root = root.right
    return closest
##Time complexity O(logN) and space complexity O(1)


##272. Closest Binary Search Tree Value II

# Given a non-empty binary search tree and a target value, find k values in the BST that are closest to the target.
# Note:
# Given target value is a floating point.
# You may assume k is always valid, that is: k ≤ total nodes.
# You are guaranteed to have only one unique set of k values in the BST that are closest to the target.
# Example:
# Input: root = [4,2,5,1,3], target = 3.714286, and k = 2
#
#     4
#    / \
#   2   5
#  / \
# 1   3
# Output: [4,3]
# Follow up:
# Assume that the BST is balanced, could you solve it in less than O(n) runtime (where n = total nodes)?

def closestValue2(root, target, k):
    res = inorder_helper(root, target, k)
    return res

def inorder_helper(root, target, k):
    if root is None:
        return None
    res = collections.queue()
    stack = []
    while root or stack:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        ##
        if len(res) == k:
            if abs(target - root.val) < abs(target - res[0]):
                res.popleft()
            else:
                return res
        res.push(root.val)
        root = root.right
    return res


##Time complexity O(n) and space complexity O(n)


##1150. Check If a Number Is Majority Element in a Sorted Array

# Description
# Given an array nums sorted in non-decreasing order, and a number target, return True if and only if target is a
# majority element.
# A majority element is an element that appears more than N/2 times in an array of length N.
# Example 1:
# Input: nums = [2,4,5,5,5,5,5,6,6], target = 5
# Output: true
# Explanation:
# The value 5 appears 5 times and the length of the array is 9.
# Thus, 5 is a majority element because 5 > 9/2 is true.
# Example 2:
# Input: nums = [10,100,101,101], target = 101
# Output: false
# Explanation:
# The value 101 appears 2 times and the length of the array is 4.
# Thus, 101 is not a majority element because 2 > 4/2 is false.


def isMajorityElements(nums, target):
    i = 0
    while i < len(nums) and nums[i] != target:
        i += 1
    j = i
    while j < len(nums) and nums[j] == target:
        j += 1
    if j - i > len(nums)/2:
        return True
    else:
        return False


# nums = [10,100,101,101]
# target = 101
# print(isMajorityElements(nums, target))


##249. Group Shifted Strings
# Given a string, we can "shift" each of its letter to its successive letter, for example: "abc" -> "bcd".
# We can keep "shifting" which forms the sequence:
#
# "abc" -> "bcd" -> ... -> "xyz"
# Given a list of non-empty strings which contains only lowercase alphabets, group all strings that belong to
# the same shifting sequence.
# Example:
# Input:
# ["abc", "bcd", "acef", "xyz", "az", "ba", "a", "z"],
# Output:
# [
#   ["abc","bcd","xyz"],
#   ["az","ba"],
#   ["acef"],
#   ["a","z"]
# ]

def groupShiftedStr(strings):
    res = {}
    for string in strings:
        key = ""
        for char in range(1, len(string)):
            key += str(((ord(string[char]) - ord(string[char-1]))+26)%26)
        if key in res:
            res[key].append(string)
        else:
            res[key] = [string]
    return list(res.values())

# strings = ["abc", "bcd", "acef", "xyz", "az", "ba", "a", "z"]
# print(groupShiftedStr(strings))


# 266. Palindrome Permutation
# Given a string, determine if a permutation of the string could form a palindrome.
# Example 1:
# Input:
# "code"
# Output: false
# Example 2:
# Input:
# "aab"
# Output: true
# Hints
# Consider the palindromes of odd vs even length. What difference do you notice?
# Count the frequency of each character.
# If each character occurs even number of times, then it must be a palindrome. How about character which occurs
# odd number of times?

##if string is pelindrom then atmost one character is odd number count

def pelinPermutation(s):
    dictS = {}
    for c in s:
        if c in dictS:
            dictS[c] += 1
        else:
            dictS[c] = 1

    count = 0
    for key in dictS:
        count += dictS[key] % 2
    return count <= 1

##Time complexity O(n) and space complexity O(n)
# s = "carerac"
# print(pelinPermutation(s))


##251. Flatten 2D Vector
# Design and implement an iterator to flatten a 2d vector. It should support the following operations: next and hasNext.
# Example:
#
# Vector2D iterator = new Vector2D([[1,2],[3],[4]]);
# iterator.next(); // return 1
# iterator.next(); // return 2
# iterator.next(); // return 3
# iterator.hasNext(); // return true
# iterator.hasNext(); // return true
# iterator.next(); // return 4
# iterator.hasNext(); // return false
# Notes:
#
# Please remember to RESET your class variables declared in Vector2D, as static/class variables are persisted
# across multiple test cases. Please see here for more details.
# You may assume that next() call will always be valid, that is, there will be at least a next element in the 2d
# vector when next() is called.


class vector2D:
    def __init__(self, v):
        self.vector = v
        self.outer = 0
        self.inner = 0

    def advance_to_next(self):
        while self.outer < len(self.vector) and self.inner == len(self.vector[self.outer]):
            self.outer += 1
            self.inner = 0

    def next(self):
        self.advance_to_next()
        res = self.vector[self.outer][self.inner]
        self.inner += 1
        return res
    def hasNext(self):
        self.advance_to_next()
        return self.outer < len(self.vector)


##Time complexity-->
# Constructor - O(1)
# Let N be the number of integers within the 2D Vector, and V be the number of inner vectors.
# advance_to_next -- O(v/n)
# next() / hasNext -- O(1)
# space - O(1)
# vect = vector2D([[1,2],[3],[4]])
# print(vect.next())
# print(vect.next())
# print(vect.next())
# print(vect.hasNext())
# print(vect.hasNext())
# print(vect.next())
# print(vect.hasNext())

##348. Design Tic-Tac-Toe
# Assume the following rules are for the tic-tac-toe game on an n x n board between two players:
# A move is guaranteed to be valid and is placed on an empty block.
# Once a winning condition is reached, no more moves are allowed.
# A player who succeeds in placing n of their marks in a horizontal, vertical, or diagonal row wins the game.
# Implement the TicTacToe class:
# TicTacToe(int n) Initializes the object the size of the board n.
# int move(int row, int col, int player) Indicates that player with id player plays at the cell (row, col) of the
# board. The move is guaranteed to be a valid move.
# Follow up:
# Could you do better than O(n2) per move() operation?

# Input
# ["TicTacToe", "move", "move", "move", "move", "move", "move", "move"]
# [[3], [0, 0, 1], [0, 2, 2], [2, 2, 1], [1, 1, 2], [2, 0, 1], [1, 0, 2], [2, 1, 1]]
# Output
# [null, 0, 0, 0, 0, 0, 0, 1]

class TicTacToe:
    def __init__(self, n):
        self.s = n
        self.r = [0]*n
        self.c = [0]*n
        self.diag1 = 0
        self.diag2 = 0

    def move(self, row, col, p):
        if p == 1:
            self.r[row] += 1
            self.c[col] += 1
            if row == col:
                self.diag1 += 1
            if row + col == self.s - 1:
                self.diag2 += 1
            if self.r[row] == self.s or self.c == self.s or self.diag1 == self.s or self.diag2 == self.s:
                return 1
            else:
                return 0
        if p == 2:
            self.r[row] -= 1
            self.c[col] -= 1
            if row == col:
                self.diag1 -= 1
            if row + col == self.s - 1:
                self.diag2 -= 1
            if self.r[row] == -self.s or self.c == -self.s or self.diag1 == -self.s or self.diag2 == -self.s:
                return 2
            else:
                return 0

# tic = TicTacToe(3)
# print(tic.move(0, 0, 1))
# print(tic.move(0, 2, 2))
# print(tic.move(2, 2, 1))
# print(tic.move(1, 1, 2))
# print(tic.move(2, 0, 1))
# print(tic.move(1, 0, 2))
# print(tic.move(2, 1, 1))


##1245. Tree Diameter
# Given an undirected tree, return its diameter: the number of edges in a longest path in that tree.
# The tree is given as an array of edges where edges[i] = [u, v] is a bidirectional edge between nodes u and v.
# Each node has labels in the set {0, 1, ..., edges.length}.
# Input: edges = [[0,1],[0,2]]
# Output: 2
# Explanation:
# A longest path of the tree is the path 1 - 0 - 2.
#
# Input: edges = [[0,1],[1,2],[2,3],[1,4],[4,5]]
# Output: 4
# Explanation:
# A longest path of the tree is the path 3 - 2 - 1 - 4 - 5.


def treeDiameter(edges):
    import collections
    if not edges:
        return 0
    n = len(edges) + 1
    adj_list = {}
    for node in range(n+1):
        adj_list[node] = []
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)

    def bfs(source):
        queue = collections.deque([[source, 0]])
        seen = set()
        while len(queue) != 0:
            node, d = queue.popleft()
            for neighbour in adj_list[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    queue.append([neighbour, d+1])
        return node, d
    farthest, d1 = bfs(0)
    farthest2, d2 = bfs(farthest)
    return d2

# edges = [[0,1],[0,2]]
# print(treeDiameter(edges))

##408. Valid Word Abbreviation
# Given a non-empty string s and an abbreviation abbr, return whether the string matches with the given abbreviation.
# A string such as "word" contains only the following valid abbreviations:
# ["word", "1ord", "w1rd", "wo1d", "wor1", "2rd", "w2d", "wo2", "1o1d", "1or1", "w1r1", "1o2", "2r1", "3d", "w3", "4"]
# Notice that only the above abbreviations are valid abbreviations of the string "word". Any other string is not a
# valid abbreviation of "word".
# Note:
# Assume s contains only lowercase letters and abbr contains only lowercase letters and digits.
# Example 1:
# Given s = "internationalization", abbr = "i12iz4n":
# Return true.

def validWordAbbreviation(word, abbr):
        i = 0
        j = 0
        while i < len(word) and j < len(abbr):
            if word[i] == abbr[j]:
                i += 1
                j += 1
            elif abbr[j] == '0':
                return False
            elif abbr[j].isdigit():
                k = j
                while k < len(abbr) and abbr[k].isdigit():
                    k += 1
                i += int(abbr[j:k])
                j = k
            else:
                return False
        return i == len(word) and j == len(abbr)

# word = "apple"
# abbr = "a2e"
# print(validWordAbbreviation(word, abbr))


##271. Encode and Decode Strings
# Design an algorithm to encode a list of strings to a string. The encoded string is then sent over the network
# and is decoded back to the original list of strings.
# Machine 1 (sender) has the function:
#
# string encode(vector<string> strs) {
#   // ... your code
#   return encoded_string;
# }
# Machine 2 (receiver) has the function:
# vector<string> decode(string s) {
#   //... your code
#   return strs;
# }
# So Machine 1 does:
# string encoded_string = encode(strs);
# and Machine 2 does:
# vector<string> strs2 = decode(encoded_string);
# strs2 in Machine 2 should be the same as strs in Machine 1.
# Implement the encode and decode methods.

# Note:
# The string may contain any possible characters out of 256 valid ascii characters. Your algorithm should be
# generalized enough to work on any possible characters.
# Do not use class member/global/static variables to store states. Your encode and decode algorithms should be
# stateless.
# Do not rely on any library method such as eval or serialize methods. You should implement your own encode/decode
# algorithm.

class encodeAndDecode:     ##   4#abcd

    def encode(self, strs):
        res = ""
        for s in strs:
            res += str(len(s)) + "#" + s
        return res

    def decode(self, s):
        res = []
        i = 0
        while i < len(s):
            j = i
            while s[j] != "#":
                j += 1
            length = int(s[i:j])
            res.append(s[j+1:j+1+length])
            i = j + 1 + length
        return res

# ende = encodeAndDecode()
# print(ende.encode(["abc","bcd","&#$aaa4"]))
# s = ende.encode(["abc","bcd","&#$aaa4"])
# print(ende.decode(s))



#1119. Remove Vowels from a String
# Given a string S, remove the vowels 'a', 'e', 'i', 'o', and 'u' from it, and return the new string.
#
# Example 1:
# Input: "leetcodeisacommunityforcoders"
# Output: "ltcdscmmntyfrcdrs"
# Example 2:
# Input: "aeiou"
# Output: ""
# Note:
# S consists of lowercase English letters only.
# 1 <= S.length <= 1000
# Hints
# How to erase vowels in a string?
# Loop over the string and check every character, if it is a vowel ignore it otherwise add it to the answer.


def removeVowls(s):
    vSet = set()
    vSet.add("a")
    vSet.add("e")
    vSet.add("i")
    vSet.add("o")
    vSet.add("u")

    res = ""
    for c in s:
        if c not in vSet:
            res += c
    return res

# s = "aeiou"
# print(removeVowls(s))


###Evaluation of Expression Tree(1628)
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def evalTree(root):
    if root is None:
        return 0
    if root.left is None and root.right is None:
        return int(root.val)
    leftSum = evalTree(root.left)
    rightSum = evalTree(root.right)

    if root.val == "+":
        return leftSum + rightSum
    elif root.val == "-":
        return leftSum - rightSum
    elif root.val == "*":
        return leftSum*rightSum
    else:
        return leftSum/rightSum

# root = Node('+')
# root.left = Node('*')
# root.left.left = Node('5')
# root.left.right = Node('4')
# root.right = Node('-')
# root.right.left = Node('100')
# root.right.right = Node('20')
#
# print(evalTree(root))


##261	Graph Valid Tree

#  Given n nodes labeled from 0 to n - 1 and a list of undirected edges (each edge is a pair of nodes),
#  write a function to check whether these edges make up a valid tree.
#  For example:
#  Given n = 5 and edges = [[0, 1], [0, 2], [0, 3], [1, 4]], return true.
#  Given n = 5 and edges = [[0, 1], [1, 2], [2, 3], [1, 3], [1, 4]], return false.
#  Hint:
#  Given n = 5 and edges = [[0, 1], [1, 2], [3, 4]], what should your return?
# Is this case a valid tree?
# According to the definition of tree on Wikipedia: “a tree is an undirected graph in which any two vertices are
# connected by exactly one path.
# In other words, any connected graph without simple cycles is a tree.”
#  Note:
#  you can assume that no duplicate edges will appear in edges.
#  Since all edges are undirected, [0, 1] is the same as [1, 0] and thus will not appear together in edges.

def graphValidTree(n, edges):
    nodes = [i for i in range(n)]
    adj_list = {}
    for node in nodes:
        adj_list[node] = []

    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)
    if not n:
        return True
    visited = set()
    stack = [0]
    prev = -1

    while len(stack) != 0:
        cur = stack.pop()
        visited.add(cur)
        for neighbour in adj_list[cur]:
            if neighbour == prev:
                continue
            prev = cur
            if neighbour in visited:
                return False
            else:
                stack.append(neighbour)
    return n == len(visited)

# n = 6
# edges = [[0, 1], [0, 2], [0, 3], [1, 4]]
# print(graphValidTree(n, edges))

## 1213  intersection-of-three-sorted-arrays

def intersectThreeSorted(num1, nums2, nums3):
    i = 0
    j = 0
    k = 0
    res = []
    while i < len(nums1) and j < len(nums2) and k < len(nums3):
        if nums1[i] == nums2[j] == nums3[k]:
            res.append(nums1[i])
            i += 1
            j += 1
            k += 1
        elif nums1[i] < nums2[j]:
            i += 1
        elif nums2[j] < nums3[k]:
            j += 1
        else:
            k += 1
    return res
# nums1 = [1,2,3,4,5]
# nums2 = [1,2,5,7,9]
# nums3 = [1,3,4,5,8]
# print(intersectThreeSorted(nums1, nums2, nums3))


##490. The Maze (Medium)
# Input 1: a maze represented by a 2D array
#
# 0 0 1 0 0
# 0 0 0 0 0
# 0 0 0 1 0
# 1 1 0 1 1
# 0 0 0 0 0
#
# Input 2: start coordinate (rowStart, colStart) = (0, 4)
# Input 3: destination coordinate (rowDest, colDest) = (4, 4)
#
# Output: true
#
# Explanation: One possible way is : left -> down -> left -> down -> right -> down -> right.

def maze(grid, start, destination):
    stack = [(start[0], start[1])]
    seen = set()
    dist = [(1,0), (-1,0), (0,-1), (0,1)]

    while len(stack) != 0:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add((cur[0], cur[1]))
        i = cur[0]
        j = cur[1]
        for d in dist:
            row = i
            col = j
            while row >= 0 and row < len(grid) and col >= 0 and col < len(grid[0]) and grid[row][col] == 0:
                row += d[0]
                col += d[1]
            row -= d[0]
            col -= d[1]

            if row == destination[0] and col == destination[1]:
                return True
            if (row, col) not in seen:
                stack.append((row, col))
                #seen.add((row, col))
    return False

# grid = [[0,0,1,0,0],[0, 0, 0, 0, 0,], [0, 0, 0, 1, 0,],[1, 1, 0, 1, 1,],[0, 0, 0, 0, 0]]
# start = (0, 4)
# destination = (4, 4)
# print(maze(grid, start, destination))


##505. The Maze II
# Input 1: a maze represented by a 2D array
#
# 0 0 1 0 0
# 0 0 0 0 0
# 0 0 0 1 0
# 1 1 0 1 1
# 0 0 0 0 0
#
# Input 2: start coordinate (rowStart, colStart) = (0, 4)
# Input 3: destination coordinate (rowDest, colDest) = (4, 4)
#
# Output: 12
# Explanation: One shortest way is : left -> down -> left -> down -> right -> down -> right.
#              The total distance is 1 + 1 + 3 + 1 + 2 + 2 + 2 = 12.


def maze2(grid, start, destination):
    import heapq
    distance = {}
    distance[(start[0], start[1])] = 0
    heap = [(0, start[0], start[1])]
    directions = [(1,0), (-1,0), (0, 1), (0,-1)]
    while heap:
        dist, i, j = heapq.heappop(heap)
        if i == destination[0] and j == destination[1]:
            return dist
        for d in directions:
            row = i
            col = j
            steps = dist
            while row >= 0 and row < len(grid) and col >= 0 and col < len(grid[0]) and grid[row][col] == 0:
                row += d[0]
                col += d[1]
                steps += 1
            row -= d[0]
            col -= d[1]
            steps -= 1

            if (row, col) not in distance or steps < distance[(row, col)]:
                distance[(row, col)] = steps
                heapq.heappush(heap, (steps, row, col))
    return -1


# grid = [[0,0,1,0,0],[0, 0, 0, 0, 0,], [0, 0, 0, 1, 0,],[1, 1, 0, 1, 1,],[0, 0, 0, 0, 0]]
# start = (0, 4)
# destination = (4, 4)
# print(maze2(grid, start, destination))

#499. The Maze III
# Input 1: a maze represented by a 2D array
#
# 0 0 0 0 0
# 1 1 0 0 1
# 0 0 0 0 0
# 0 1 0 0 1
# 0 1 0 0 0
#
# Input 2: ball coordinate (rowBall, colBall) = (4, 3)
# Input 3: hole coordinate (rowHole, colHole) = (0, 1)
#
# Output: "lul"
#
# Explanation: There are two shortest ways for the ball to drop into the hole.
# The first way is left -> up -> left, represented by "lul".
# The second way is up -> left, represented by 'ul'.
# Both ways have shortest distance 6, but the first way is lexicographically smaller because 'l' < 'u'. So the output is "lul".


def maze3(grid, ball, hole):
    import heapq
    distance = {}
    distance[(ball[0], ball[1])] = [0, ""]

    heap = [(0, "", ball[0], ball[1])]
    directions = [(1,0,'d'), (-1,0, 'u'), (0,1, 'r'), (0,-1,'l')]

    while heap:
        dist, pattern, i, j = heapq.heappop(heap)
        if i == hole[0] and j == hole[1]:
            return pattern
        for dx, dy, p in directions:
            row = i
            col = j
            steps = dist
            while 0<=row+dx<len(grid) and 0<=col+dy<len(grid[0]) and grid[row+dx][col+dy] != 1:
                row += dx
                col += dy
                steps += 1

                if row == hole[0] and col == hole[1]:
                    break
            if (row, col) not in distance or [steps, pattern+p] < distance[(row,col)]:
                distance[(row, col)] = [steps, pattern+p]
                heapq.heappush(heap, (steps, pattern+p, row, col))
    return "impossible"

# grid = [[0, 0, 0, 0, 0],[1, 1, 0, 0, 1],[0, 0, 0, 0, 0],[0, 1, 0, 0, 1],[0, 1, 0, 0, 0]]
#
# ball = (4, 3)
# hole = (0, 1)
#
# print(maze3(grid, ball, hole))


# 323. Number of Connected Components in an Undirected Graph
# Given n nodes labeled from 0 to n - 1 and a list of undirected edges (each edge is a pair of nodes),
# write a function to find the number of connected components in an undirected graph.

# Input: n = 5 and edges = [[0, 1], [1, 2], [3, 4]]
#
#      0          3
#      |          |
#      1 --- 2    4
#
# Output: 2

def numOfComponent(n, edges):
    if len(edges) == 0:
        return n
    nodes = [i for i in range(n)]
    adj_list = {}
    for node in nodes:
        adj_list[node] = []
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)

    count = 0
    seen = set()
    for node in nodes:
        if node not in seen:
            numOfComponent_DFS(adj_list, seen, node)
            count += 1
    return count

def numOfComponent_DFS(adj_list, seen, node):
    stack = [node]

    while len(stack)!= 0:
        cur = stack.pop()
        seen.add(cur)
        for neighbour in adj_list[cur]:
            if neighbour not in seen:
                stack.append(neighbour)

# n = 5
# edges = [[0, 1], [1, 2], [2, 3], [3, 4]]
# print(numOfComponent(n, edges))


##1188. Design Bounded Blocking Queue

# Input:
# 1
# 1
# ["BoundedBlockingQueue","enqueue","dequeue","dequeue","enqueue","enqueue","enqueue","enqueue","dequeue"]
# [[2],[1],[],[],[0],[2],[3],[4],[]]
#
# Output:
# [1,0,2,2]
#
# Explanation:
# Number of producer threads = 1
# Number of consumer threads = 1
#
# BoundedBlockingQueue queue = new BoundedBlockingQueue(2);   // initialize the queue with capacity = 2.
#
# queue.enqueue(1);   // The producer thread enqueues 1 to the queue.
# queue.dequeue();    // The consumer thread calls dequeue and returns 1 from the queue.
# queue.dequeue();    // Since the queue is empty, the consumer thread is blocked.
# queue.enqueue(0);   // The producer thread enqueues 0 to the queue. The consumer thread is unblocked and returns 0 from the queue.
# queue.enqueue(2);   // The producer thread enqueues 2 to the queue.
# queue.enqueue(3);   // The producer thread enqueues 3 to the queue.
# queue.enqueue(4);   // The producer thread is blocked because the queue's capacity (2) is reached.
# queue.dequeue();    // The consumer thread returns 2 from the queue. The producer thread is unblocked and enqueues 4 to the queue.
# queue.size();       // 2 elements remaining in the queue. size() is always called at the end of each test case.


class boundedBlockingQueue:
    from threading import Lock
    import collections
    def init__(self, capacity):
        self.en = Lock()
        self.de = Lock()
        self.q = collections.deque()
        self.capacity = capacity
        self.de.acquire()

    def enque(self, element):
        self.en.acquire()
        self.q.append(element)
        if self.size()< self.capacity:
            self.en.release()
        if self.de.locked():
            self.de.release()

    def deque(self):
        self.de.acquire()
        output = self.q.popleft()
        if self.size()>0:
            self.de.release()
        if self.en.locked():
            self.en.release()
        return output

    def size(self):
        return len(self.q)




