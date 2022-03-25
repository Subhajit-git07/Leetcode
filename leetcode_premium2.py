##1166. Design File System
# Input:
# ["FileSystem","createPath","createPath","get","createPath","get"]
# [[],["/leet",1],["/leet/code",2],["/leet/code"],["/c/d",1],["/c"]]
# Output:
# [null,true,true,2,false,-1]
# Explanation:
# FileSystem fileSystem = new FileSystem();
#
# fileSystem.createPath("/leet", 1); // return true
# fileSystem.createPath("/leet/code", 2); // return true
# fileSystem.get("/leet/code"); // return 2
# fileSystem.createPath("/c/d", 1); // return false because the parent path "/c" doesn't exist.
# fileSystem.get("/c"); // return -1 because this path doesn't exist.

class TrieNode:
    def __init__(self):
        self.val = None
        self.children = {}

# class Trie:
#     def __init__(self):
#         self.root = TrieNode(None)

class FileSystem:
    def __init__(self):
        self.root = TrieNode()


    def createPath(self, path, value):
        parent = self.root
        path = path.split('/')
        for i in range(1, len(path)-1):
            if path[i] not in parent.children:
                return False
            parent = parent.children[path[i]]
        last = path[len(path) - 1]
        if last in parent.children:
            return False
        parent.children[last] = TrieNode()
        parent = parent.children[last]
        parent.val = value
        return True

    def get(self, path):
        parent = self.root
        path = path.split('/')
        for i in range(1, len(path)):
            if path[i] not in parent.children:
                return -1
            parent = parent.children[path[i]]
        return parent.val

# FS = FileSystem()
# print(FS.createPath("/leet", 1))
# print(FS.createPath("/leet/code", 2))
# print(FS.get("/leet/code"))
# print(FS.createPath("/c/d", 1))
# print(FS.get("/c"))

# path = "/leet"
# path = path.split('/')
# print(path)
# for i in range(1, len(path)-1):
#     print(path[i], "----")

##635. Design Log Storage System

# put(1, "2017:01:01:23:59:59");
# put(2, "2017:01:01:22:59:59");
# put(3, "2016:01:01:00:00:00");
# retrieve("2016:01:01:01:01:01","2017:01:01:23:00:00","Year"); // return [1,2,3], because you need to return all
# logs within 2016 and 2017.
# retrieve("2016:01:01:01:01:01","2017:01:01:23:00:00","Hour"); // return [1,2], because you need to return all logs
# start from 2016:01:01:01 to 2017:01:01:23, where log 3 is left outside the range.

class LogStorage:
    def __init__(self):
        self.logs = []

    def put(self, id, timeStamp):
        self.logs.append((id, timeStamp))

    def retrieve(self, s, e, gra):
        index = {"Year":5, "Month":8, "Day":11, "Hour":14, "Minute":17, "Second":20}[gra]
        start = s[:index]
        end = e[:index]

        return [tid for tid, timeStamp in self.logs if start <= timeStamp[:index] <= end]

# log = LogStorage()
# log.put(1, "2017:01:01:23:59:59")
# log.put(2, "2017:01:01:22:59:59")
# log.put(3, "2016:01:01:00:00:00")
# print(log.retrieve("2016:01:01:01:01:01","2017:01:01:23:00:00","Year"))
# print(log.retrieve("2016:01:01:01:01:01","2017:01:01:23:00:00","Hour"))

##1136. Parallel Courses
# Input: N = 3, relations = [[1,3],[2,3]]
# Output: 2
# Explanation:
# In the first semester, courses 1 and 2 are studied. In the second semester, course 3 is studied.

def parallelCourse(n, relations):
    adj_list = {}
    inDegree = {}

    for i in range(1, n+1):
        adj_list[i] = []
    for i in range(1, n+1):
        inDegree[i] = 0

    for i in relations:
        preRequisite = i[0]
        course = i[1]
        adj_list[preRequisite].append(course)
        inDegree[course] += 1
    print(inDegree)

    import collections
    queue = collections.deque([i for i in range(1, n+1) if inDegree[i]==0])
    print(len(queue))
    res = 0
    totalCourse = 0

    while len(queue) != 0:
        res += 1
        for _ in range(len(queue)):
            cur = queue.popleft()
            totalCourse += 1
            for neighbour in adj_list[cur]:
                inDegree[neighbour] -=1
                if inDegree[neighbour] == 0:
                    queue.append(neighbour)

    if totalCourse == n:
        return res
    else:
        return -1

# n = 3
# relations = [[1,2],[2,3],[3,1]]
# print(parallelCourse(n, relations))

##604. Design Compressed String Iterator
# Design and implement a data structure for a compressed string iterator. It should support the following operations:
# next and hasNext.
#
# The given compressed string will be in the form of each letter followed by a positive integer representing the number
# of this letter existing in the original uncompressed string.
#
# next() - if the original string still has uncompressed characters, return the next letter; Otherwise return a white
# space.
# hasNext() - Judge whether there is any letter needs to be uncompressed.

class stringComparator:
    def __init__(self, s):
        self.newStr = ""
        for i in range(0, len(s)-1, 2):
            self.newStr += s[i]*int(s[i+1])
        self.pointer = -1

    def next(self):
        self.pointer += 1
        if self.pointer <= len(self.newStr) - 1:
            return self.newStr[self.pointer]
        else:
            return " "

    def hasNext(self):
        return self.pointer < len(self.newStr) - 1

# s = "L1e2t1C1o1d1e1"
# SC = stringComparator(s)
# print(SC.next())
# print(SC.next())
# print(SC.next())
# print(SC.next())
# print(SC.next())
# print(SC.next())
# print(SC.next())
# print(SC.hasNext())
# print(SC.next())
# print(SC.hasNext())
# print(SC.next())


##379. Design Phone Directory
# Design a Phone Directory which supports the following operations:
#
# get: Provide a number which is not assigned to anyone.
# check: Check if a number is available or not.
# release: Recycle or release a number.

class phoneDict:
    def __init__(self, n):
        self.n = n
        self.available = set()
        for i in range(n):
            self.available.add(i)

    def get(self):
        res = -1
        for num in self.available:
            res = num
            break
        self.available.remove(num)
        return res

    def check(self, num):
        if num in self.available:
            return True
        return False

    def release(self, num):
        self.available.add(num)


##362. Design Hit Counter

# Design a hit counter which counts the number of hits received in the past 5 minutes.
#
# Each function accepts a timestamp parameter (in seconds granularity) and you may assume that calls are being made to
# the system in chronological order (ie, the timestamp is monotonically increasing). You may assume that the earliest
# timestamp starts at 1.
#
# It is possible that several hits arrive roughly at the same time.





























##252. Meeting Rooms

# Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],...]
# (si < ei), determine if a person could attend all meetings.
# Input: [[0,30],[5,10],[15,20]]
# Output: false

def meetingRoom(intervals):
    intervals.sort(key=lambda x:x[0])
    for i in range(1, len(intervals)):
        start = intervals[i][0]
        prevEnd = intervals[i-1][1]
        if start < prevEnd:
            return False
    return True

##Time O(NlogN)
# intervals = [[7,10],[2,4]]
# print(meetingRoom(intervals))


##253. Meeting Rooms II

# Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],...]
# (si < ei), find the minimum number of conference rooms required.
# Input: [[0, 30],[5, 10],[15, 20]]
# Output: 2

def meetingRoom2(intervals):
    intervals.sort(key=lambda x:x[0])
    print(intervals)
    import heapq
    minHeap = []
    room = 0
    for i in range(len(intervals)):
        start = intervals[i][0]
        end = intervals[i][1]
        if len(minHeap) != 0 and start >= minHeap[0]:
            heapq.heappop(minHeap)
        else:
            room += 1
        heapq.heappush(minHeap, end)
    return room

##Time O(NlogN)
# intervals = [[7,10],[2,4], [1,5],[10,30],[11,26],[30,40]]
# print(meetingRoom2(intervals))


##353. Design Snake Game
import collections
class snake:

    def __init__(self, width, height, food):
        self.width = width
        self.height = height
        self.food = food
        self.queue = collections.deque([])
        self.queue.append((0,0))
        self.score = 0


    def move(self, step):
        r, c = self.queue[-1]
        if step == "L":
            c = c - 1
        elif step == "R":
            c = c + 1
        elif step == "U":
            r = r - 1
        elif step == "D":
            r = r + 1
        if r < 0 or r >=self.height or c < 0 or c >= self.width:
            return -1

        if self.food and [r, c] == self.food[0]:
            self.food.pop(0)
            self.queue.append((r, c))
            self.score += 1

        else:
            self.queue.popleft()
            if (r, c) in self.queue:
                return -1
            else:
                self.queue.append((r, c))

        return self.score

# width = 3
# height = 2
# food = [[1,2],[0,1]]
# snakeGame = snake(width, height, food)
# print(snakeGame.move("R"))
# print(snakeGame.move("D"))
# print(snakeGame.move("R"))
# print(snakeGame.move("U"))
# print(snakeGame.move("L"))
# print(snakeGame.move("U"))


###1586. Binary Search Tree Iterator II
# Explanation
# // The underlined element is where the pointer currently is.
# BSTIterator bSTIterator = new BSTIterator([7, 3, 15, null, null, 9, 20]); // state is   [3, 7, 9, 15, 20]
# bSTIterator.next(); // state becomes [3, 7, 9, 15, 20], return 3
# bSTIterator.next(); // state becomes [3, 7, 9, 15, 20], return 7
# bSTIterator.prev(); // state becomes [3, 7, 9, 15, 20], return 3
# bSTIterator.next(); // state becomes [3, 7, 9, 15, 20], return 7
# bSTIterator.hasNext(); // return true
# bSTIterator.next(); // state becomes [3, 7, 9, 15, 20], return 9
# bSTIterator.next(); // state becomes [3, 7, 9, 15, 20], return 15
# bSTIterator.next(); // state becomes [3, 7, 9, 15, 20], return 20
# bSTIterator.hasNext(); // return false
# bSTIterator.hasPrev(); // return true
# bSTIterator.prev(); // state becomes [3, 7, 9, 15, 20], return 15
# bSTIterator.prev(); // state becomes [3, 7, 9, 15, 20], return 9


class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


class BST_iteratorII:
    def __init__(self, root):
        self.pointer = -1
        self.stack = []
        self.output = []
        self.partialInorder(root)

    def partialInorder(self, root):
        while root:
            self.stack.append(root)
            root = root.left

    def next(self):
        self.pointer += 1
        if self.pointer >= 0 and self.pointer < len(self.output):
            res = self.output[self.pointer]
            return res.val
        else:
            res = self.stack.pop()
            self.partialInorder(res.right)
            self.output.append(res)
            return res.val

    def hasNext(self):
        if self.pointer + 1 >= 0 and self.pointer + 1 < len(self.output):
            return True
        return len(self.stack) != 0

    def prev(self):
        self.pointer -= 1
        return self.output[self.pointer].val

    def hasPrev(self):
        if self.pointer - 1 >= 0 and self.pointer - 1 < len(self.output):
            return True
        return False


# root = TreeNode(7)
# root.left = TreeNode(3)
# root.right = TreeNode(15)
# root.right.left = TreeNode(9)
# root.right.right = TreeNode(20)
#
# bst_iterator2 = BST_iteratorII(root)
# print(bst_iterator2.next())
# print(bst_iterator2.next())
# print(bst_iterator2.prev())
#
# print(bst_iterator2.next())
# print(bst_iterator2.hasNext())
# print(bst_iterator2.next())
# print(bst_iterator2.next())
# print(bst_iterator2.next())
#
# print(bst_iterator2.hasNext())
# print(bst_iterator2.hasPrev())
# print(bst_iterator2.prev())
# print(bst_iterator2.prev())

##1135. Connecting Cities With Minimum Cost
def minCostCities(n, connections):
    graph = []
    for u,v,w in connections:
        graph.append((u,v,w))

    parent = []
    rank = []
    for i in range(1, n+1):
        parent.append(i)
        rank.append(0)
    i = 0
    edge = 0
    graph = sorted(graph, key = lambda x:x[2])
    weight = 0

    while i < len(graph):
        u, v, w = graph[i]
        i += 1
        u_parent = find(parent, u)
        v_parent = find(parent, v)
        if u_parent != v_parent:
            union(parent, rank, u_parent, v_parent)
            edge += 1
            weight += w

    if edge != n - 1:
        return -1
    else:
        return weight


def find(parent, node):
    if node != parent[node-1]:
        node = find(parent, parent[node-1])
    return parent[node-1]


def union(parent, rank, u, v):
    u_parent = find(parent, u)
    v_parent = find(parent, v)

    if rank[u_parent-1] > rank[v_parent-1]:
        parent[v_parent-1] = u_parent
    elif rank[u_parent-1] < rank[v_parent-1]:
        parent[u_parent-1] = v_parent
    else:
        parent[v_parent-1] = u_parent
        rank[u_parent] += 1

# n = 3
# connections = [[1,2,5],[1,3,6],[2,3,1]]
# print(minCostCities(n, connections))



####161. One Edit Distance
# Input: s = "ab", t = "acb"
# Output: true
# Explanation: We can insert 'c' into s to get t.

def oneEditDistance(s, t):
    if abs(len(s) - len(t)) > 1:
        return False
    i = 0
    j = 0
    count = 0
    while i < len(s) and j < len(t):
        if s[i] != t[j]:
            if count == 1:
                return False
            if len(s) > len(t):
                i += 1
            elif len(s) < len(t):
                j += 1
            else:
                i += 1
                j += 1
            count += 1
        else:
            i += 1
            j += 1

    if i < len(s) or j < len(t):
        count += 1

    return count == 1

# s = "cab"
# t = "ad"
# print(oneEditDistance(s, t))

###277. Find the Celebrity

# Input: graph = [
#   [1,1,0],
#   [0,1,0],
#   [1,1,1]
# ]
# Output: 1
# Explanation: There are three persons labeled with 0, 1 and 2. graph[i][j] = 1 means person i knows
# person j, otherwise graph[i][j] = 0 means person i does not know person j. The celebrity is the person
# labeled as 1 because both 0 and 2 know him but 1 does not know anybody.

def findCelebrity(graph):
    n = len(graph)
    candidate = 0
    for i in range(1, n):
        if graph[candidate][i] == 1:
            candidate = i

    for i in range(n):
        if (i != candidate & graph[candidate][i] == 1) or graph[i][candidate] == 0:
            return -1
    return candidate

# graph = [
#   [1,0,1],
#   [1,1,0],
#   [0,1,1]
# ]
# print(findCelebrity(graph))

###311 - Sparse Matrix Multiplication
def sparseMatmulti(A, B):
    C = [[0 for col in range(len(B[0]))] for row in range(len(A))]

    for i in range(len(C)):
        for k in range(len(A[0])):
            if A[i][k] != 0: ##non zero check, if zero then skip because of spase matrix
                for j in range(len(C[0])):
                    C[i][j] += A[i][k] * B[k][j]

    return C

# A = [
#  [ 1, 0, 0],
#  [-1, 0, 3]
#  ]
#
# B = [
#  [ 7, 0, 0 ],
#  [ 0, 0, 0 ],
#  [ 0, 0, 1 ]
#  ]
#
# print(sparseMatmulti(A, B))



###269. Alien Dictionary
# Input:
# [
#   "wrt",
#   "wrf",
#   "er",
#   "ett",
#   "rftt"
# ]
#
# Output: "wertf"


def alienDictionary(words):
    adj_list = {c:set() for w in words for c in w}
    for i in range(len(words) - 1):
        word1 = words[i]
        word2 = words[i+1]
        minLen = min(len(word1), len(word2))
        if len(word1) > len(word2) and word1[:minLen] == word2[:minLen]:
            return ""
        for j in range(minLen):
            if word1[j] != word2[j]:
                adj_list[word1[j]].add(word2[j])
                break
    visited = {}
    res = []

    def dfs(node):
        if node in visited:
            return visited[node]
        visited[node] = True
        for nei in adj_list[node]:
            if dfs(nei):
                return True
        visited[node] = False
        res.append(node)

    for node in adj_list:
        if dfs(node):
            return ""
    res = res[::-1]
    return "".join(res)


# words =  [
#   "z",
#   "x",
#   "z"
# ]
# print(alienDictionary(words))
#


##325 - Maximum Size Subarray Sum Equals k

def maxSubArraySumK(nums, k):
    cSum = 0
    maxLen = 0
    sumDict = {0:-1}

    for i, num in enumerate(nums):
        cSum += num
        if cSum - k in sumDict:
            maxLen = max(maxLen, i-sumDict[cSum-k])
        if cSum not in sumDict:
            sumDict[cSum] = i
    return maxLen

##Time O(n) and space O(n)
# nums = [-2, -1, 2, 1]
# k = 1
# print(maxSubArraySumK(nums, k))


###285. Inorder Successor in BST
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def inorderSuccessor_usingInorderTraversal(root, p):
    if root is None:
        return None
    prev = None
    stack = []
    while root or stack:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        if prev and p == prev.val:
            return root.val
        prev = root
        root = root.right
    return None

##Time O(n)


def inorderSuccessor(root, p):
    if root is None:
        return None
    prev = None
    cur = root
    while cur:
        if cur.val > p:
            prev = cur
            cur = cur.left
        else:
            cur = cur.right
    return prev.val

#Time O(H)

def inorderPredesesor(root, p):
    if root is None:
        return None
    prev = None
    cur = root
    while cur:
        if cur.val < p:
            prev = cur
            cur = cur.right
        else:
            cur = cur.left
    return prev.val


# root = TreeNode(5)
# root.left = TreeNode(3)
# root.right = TreeNode(6)
# root.left.left = TreeNode(2)
# root.left.right = TreeNode(4)
# root.left.left.left = TreeNode(1)
#
#
# print(inorderPredesesor(root, 4))\

###510. Inorder Successor in BST II

class TreeNode2:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None

def inorderSuccessor2(node):
    if node is None:
        return None
    if node.right:
        node = node.right
        while node.left:
            node = node.left
        return node
    else:
        while node.parent and node.val > node.parent.val:
            node = node.parent
        if node.parent is None:
            return None
        else:
            return node.parent


##157. Read N Characters Given Read4

def read(buf, n):
    copied_char = 0
    buf4 = [""]*4
    count = 0

    while count < n:
        x = read4(buf4)
        count += 4
        if copied_char + x <= n:
            buf[copied_char:copied_char+x] = buf4
            copied_char += x
        else:
            temp = n - copied_char
            buf[copied_char:copied_char+temp] = buf4[0:temp]
            copied_char += temp
    return copied_char


##286. Walls and Gates

def wallsGate(grid):
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 0:
                wallsGate_DFS(i, j, grid, 0)
    return grid

def wallsGate_DFS(i, j, grid, count):
    if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] < count:
        return
    grid[i][j] = count
    wallsGate_DFS(i+1, j, grid, count+1)
    wallsGate_DFS(i-1, j, grid, count+1)
    wallsGate_DFS(i, j+1, grid, count+1)
    wallsGate_DFS(i, j-1, grid, count+1)

# INF = float('inf')
# grid = [[INF ,-1 ,0 ,INF],
#         [INF, INF,INF, -1],
#         [INF, -1,INF, -1],
#         [0, -1,INF, INF]]
#
# print(wallsGate(grid))

##314. Binary Tree Vertical Order Traversal
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
root = TreeNode(3)
root.left = TreeNode(9)
root.right = TreeNode(20)
root.right.left = TreeNode(15)
root.right.right = TreeNode(7)

def verticalOrderBT(root):
    res = []
    if root is None:
        return res
    dict1 = {}
    verticalOrderBT_helper(root, dict1, 0, 0)
    minC = min(list(dict1.keys()))
    maxC = max(list(dict1.keys()))

    for c in range(minC, maxC+1):
        col_value = sorted(dict1[c], key=lambda x:(x[0], x[1]))
        sorted_val = []
        for val in col_value:
            sorted_val.append(val[1])
        res.append(sorted_val)
    return res

##col:(row, val)
def verticalOrderBT_helper(root, dict1, col, row):
    if root is None:
        return
    if col in dict1:
        dict1[col].append((row, root.val))
    else:
        dict1[col] = [(row, root.val)]
    verticalOrderBT_helper(root.left, dict1, col-1, row+1)
    verticalOrderBT_helper(root.right, dict1, col+1, row+1)

# print(verticalOrderBT(root))


## 734. Sentence Similarity

def sentenceSimilarity(words1, words2, pairs):
    wordSet = set()
    words1 = words1.split(" ")
    words2 = words2.split(" ")
    if len(words1) != len(words2):
        return False
    for pair in pairs:
        wordSet.add(pair[0]+"#"+pair[1])

    for i in range(len(words1)):
        if words1[i] != words2[i] and words1[i]+"#"+words2[i] not in wordSet and words2[i]+"#"+words1[i] not in wordSet:
            return False
    return True

# words1 = "great acting skills"
# words2 = "fine drama talent"
# pairs = [["great", "fine"], ["acting","drama"], ["skills","talent"]]
# print(sentenceSimilarity(words1, words2, pairs))
##Time O(N) space O(N)

###737 - Sentence Similarity II

def sentenceSimilarity2(words1, words2, pairs):
    if len(words1) != len(words2):
        return False
    visited = set()
    graph = {}
    for pair in pairs:
        if pair[0] not in graph:
            graph[pair[0]] = set()
            graph[pair[0]].add(pair[1])
        if pair[1] not in graph:
            graph[pair[1]] = set()
            graph[pair[1]].add(pair[0])

    for i in range(len(words1)):
        if not similarityDFS(words1[i], words2[i], graph, visited):
            return False
    return True

def similarityDFS(word1, word2, graph, visited):
    for similar in graph[word2]:
        if similar in visited:
            continue
        if similar == word1:
            return True
        else:
            visited.add(similar)
            if similarityDFS(word1, similar, graph, visited):
                return True
    return False


# words1 = ["great", "acting", "skills"]
# words2 = ["fine", "drama", "talent"]
# pairs = [["great", "good"], ["fine", "good"], ["acting","drama"], ["skills","talent"]]
# print(sentenceSimilarity2(words1, words2, pairs))

# Time O(NP) --> N - max length of words1 and words2  and P --length of pairs
# space = O(N)

##1570 - Dot Product of Two Sparse Vectors

class sparseVector:
    def __init__(self, nums):
        self.dict1 = {}
        for i in range(len(nums)):
            if nums[i] != 0:
                self.dict1[i] = nums[i]

    def dotProduct(self, vec):
        res = 0
        for k, v in self.dict1.items():
            if k in vec.dict1:
                res += v*vec.dict1[k]
        return res

# nums1 = [1,0,0,2,3]
# nums2 = [0,3,0,4,0]
# v1 = sparseVector(nums1)
# v2 = sparseVector(nums2)
#
# print(v1.dotProduct(v2))


##426. Convert Binary Search Tree to Sorted Doubly Linked List
class TreeNode:
    def __init__(self, val=None):
        self.val = val
        self.left = None
        self.right = None

# class DLL:
#     def __init__(self, val):
#         self.val = val
#         self.prev = None
#         self.next = None

def BST_to_DLL(root):
    if root is None:
        return None
    if root.left is None and root.right is None:
        root.left = root
        root.right = root
        return root
    head = root
    tail = root
    while head.left != None:
        head = head.left
    while tail.right != None:
        tail = tail.right
    node = root
    prev = None
    inOrder(node, prev)
    head.left = tail
    tail.right = head
    return head

def inOrder(node, prev):
    if node is None:
        return
    if node:
        inOrder(node.left, prev)
        node.left = prev
        if prev:
            prev.right = node
        prev = node
        inOrder(node.right, prev)


###489. Robot Room Cleaner

class robot:
    def move(self):
        pass

    def turnLeft(self):
        pass

    def turnRight(self):
        pass

    def clean(self):
        pass


def cleanRoom(robot):
    seen = set()
    directions = [(-1,0), (0,1), (-1,0), (0,-1)]
    def goBack():
        robot.turnRight()
        robot.turnRight()
        robot.clean()
        robot.turnRight()
        robot.turnRight()

    def backTrack(x, y, d):
        robot.clean()

        for i in range(4):
            new_d = (d+i)%4
            new_x = x + directions[new_d][0]
            new_y = y + directions[new_d][1]

            if (new_x, new_y) not in seen and robot.move():
                seen.add(new_x, new_y)
                backTrack(new_x, new_y, new_d)
                goBack()
            robot.turnRight()
    seen.add((0,0))
    backTrack(0,0,0)

##Time complexity O(4^n) n -- number of 1 's in room/grid
## Space complexity O(n)


###708 - Insert into a Sorted Circular Linked List

class Node:
    def __init__(self, val):
        self.val = val
        self.next = None

def insert_sortedCircularLST(head, insertVal):
    node = Node(insertVal)
    if head is None:
        node.next = node
        return node
    cur = head
    while cur.next != head:
        curVal = cur.val
        curNextVal = cur.next.val

        if insertVal >= curVal and insertVal <= curNextVal:
            break
        elif curVal > curNextVal:
            if curVal >= insertVal and curNextVal >= insertVal:
                break
            if curVal <= insertVal and curNextVal <= insertVal:
                break
        cur = cur.next

    temp = cur.next
    cur.next = node
    node.next = temp
    return head


##339. Nested List Weight Sum










