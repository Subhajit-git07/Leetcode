##13. Roman to Integer
# Input: s = "IX"
# Output: 9

def romanToInt(s):
    symbolDict = {"I":1, "V":5, "X":10, "L":50, "C":100, "D":500, "M":1000}
    res = symbolDict[s[-1]]
    for i in range(len(s)-2, -1, -1):
        if symbolDict[s[i]] < symbolDict[s[i+1]]:
            res -= symbolDict[s[i]]
        else:
            res += symbolDict[s[i]]
    return res

# s = "MCMXCIV"
# print(romanToInt(s))


##12. Integer to Roman
# Input: num = 58
# Output: "LVIII"
# Explanation: L = 50, V = 5, III = 3.

def integerToRoman(num):
    symbolDict = {1000:"M", 900:"CM", 500:"D", 400:"CD", 100:"C",
                  90:"XC", 50:"L", 40:"XL", 10:"X", 9:"IX", 5:"V", 4:"IV", 1:"I"}

    res = ""
    for key in symbolDict:
        while num >= key:
            res += symbolDict[key]
            num -= key
    return res

##Time O(len of result string)  space O(1)
# num = 1994
# print(integerToRoman(num))


##283. Move Zeroes
# Input: nums = [0,1,0,3,12]
# Output: [1,3,12,0,0]

def moveZeros(nums):
    n = len(nums)
    left = 0
    right = 0

    while right < n:
        if nums[right] == 0:
            right += 1
        else:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right += 1
    return nums

# nums = [0,1,0,3,12]
# print(moveZeros(nums))

##173. Binary Search Tree Iterator
class BST_iterator:
    def __init__(self, root):
        self.stack = []
        self.partialInorder(root)

    def partialInorder(self, root):
        while root:
            self.stack.append(root)
            root = root.left

    def next(self):
        res = self.stack.pop()
        self.partialInorder(res.right)
        return res.val

    def hasNext(self):
        return len(self.stack) != 0

##Time O(1) average  space O(h)


##128. Longest Consecutive Sequence

# Input: nums = [100,4,200,1,3,2]
# Output: 4
# Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.


def longestConSequence(nums):
    maxLen = 0

    if len(nums) == 0:
        return 0
    numSet = set(nums)
    for num in numSet:
        if num - 1 not in numSet:
            curNum = num
            curLen = 1
            while curNum + 1 in numSet:
                curNum = curNum + 1
                curLen += 1
            maxLen = max(maxLen, curLen)
    return maxLen

# nums = [0,3,7,2,5,8,4,6,0,1]
# print(longestConSequence(nums))


##69. Sqrt(x)
def squareRoot(x):
    left = 0
    right = x

    while left <= right:
        mid = left + (right-left)//2
        if mid*mid == x:
            return mid
        if mid*mid > x:
            right = mid - 1
        else:
            left = mid + 1
    return left - 1

# x = 8
# print(squareRoot(x))

##43. Multiply Strings
# Input: num1 = "123", num2 = "456"
# Output: "56088"

def multiplyStr(num1, num2):
    if num1 == "0" or num2 == "0":
        return "0"
    ans = 0
    for n in num2:
        ans = ans*10 + multi(num1, n)
    return str(ans)

def multi(num1, n):
    res = 0
    for x in num1:
        res = res*10 + (ord(x)-ord("0"))*(ord(n)-ord("0"))
    return res

# num1 = "123"
# num2 = "456"
# print(multiplyStr(num1, num2))


##79. Word Search

def wordSearch(grid, word):
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == word[0] and wordSearch_BFS(grid, i, j, 0, word):
                return True
    return False

def wordSearch_BFS(grid, i, j, count, word):
    if count == len(word):
        return True
    if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] != word[count]:
        return False
    temp = grid[i][j]
    grid[i][j] = "*"
    match = wordSearch_BFS(grid, i+1, j, count+1, word) or wordSearch_BFS(grid, i-1, j, count+1, word) or wordSearch_BFS(grid, i, j+1, count+1, word) or wordSearch_BFS(grid, i, j-1, count+1, word)
    grid[i][j] = temp
    return match

##Time O(M*N*4^L) --> L = length of word
# grid = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
# word = "ABCB"
# print(wordSearch(grid, word))

##236. Lowest Common Ancestor of a Binary Tree
def LCA_BT(root, p, q):
    if root is None or root in [p,q]:
        return root
    leftNode = LCA_BT(root.left, p, q)
    rightNode = LCA_BT(root.right, p, q)

    if leftNode and rightNode:
        return root
    elif leftNode:
        return leftNode
    else:
        return rightNode

##Time O(n) space O(n)


##127. Word Ladder
# Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
# Output: 5
# Explanation: One shortest transformation sequence is "hit" -> "hot" -> "dot" -> "dog" -> cog", which is " \
# 5 words long.

def wordLadder(beginWord, endWord, wordList):
    import collections
    queue = collections.deque([])
    wordList = set(wordList)

    if len(beginWord) == 0 or endWord not in wordList:
        return 0
    queue.append(beginWord)
    level = 1
    visited = set()
    alpha = "abcdefghijklmnopqrstuvwxyz"
    while len(queue) != 0:
        size = len(queue)
        while size != 0:
            cur = queue.popleft()
            if cur == endWord:
                return level
            for i in range(len(cur)):
                for char in alpha:
                    newWord = cur[:i] + char + cur[i+1:]
                    if newWord in wordList:
                        queue.append(newWord)
                        wordList.remove(newWord)
            size -= 1
        level += 1
    return level

# beginWord = "hit"
# endWord = "cog"
# wordList = ["hot","dot","dog","lot","log"]
# print(wordLadder(beginWord, endWord, wordList))


#125. Valid Palindrome
# Input: s = "A man, a plan, a canal: Panama"
# Output: true
# Explanation: "amanaplanacanalpanama" is a palindrome.

def validPalin(s):
    newS = ""
    for c in s:
        if c.isalpha() or c.isdigit():
            newS += c.lower()

    left = 0
    right = len(newS) - 1

    while left <= right:
        if newS[left] != newS[right]:
            return False
        left += 1
        right -= 1

    return True


# s = " "
# print(validPalin(s))

##297. Serialize and Deserialize Binary Tree
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def serializeBT(root):
    if root is None:
        return "None,"
    leftH = serializeBT(root.left)
    rightH = serializeBT(root.right)
    encode = str(root.val) + "," + leftH + rightH
    return encode

def dfs(root, res):
    if root is None:
        res.append("None,")
        return
    res.append(str(root.val)+",")
    dfs(root.left, res)
    dfs(root.right, res)

data = "1,2,None,None,3,4,None,None,5,None,None,"
def deserializeBT(data):
    data = data.split(",")
    import collections
    datalist = collections.deque(data)
    root = deserializeBT_helper(datalist)
    return root

def deserializeBT_helper(datalist):
    if datalist[0] == "None":
        datalist.popleft()
        return None
    else:
        root = TreeNode(datalist[0])
        datalist.popleft()
        root.left = deserializeBT_helper(datalist)
        root.right = deserializeBT_helper(datalist)
    return root

# root = TreeNode(1)
# root.left = TreeNode(2)
# root.right = TreeNode(3)
# root.right.left = TreeNode(4)
# root.right.right = TreeNode(5)
#
# print(serializeBT(root))
# print(deserializeBT(data))


##215. Kth Largest Element in an Array
# Input: nums = [3,2,1,5,6,4], k = 2
# Output: 5

def kth_largest(nums, k):
    left = 0
    right = len(nums) - 1

    while left <= right:
        idx = partition(left, right, nums)
        if idx == k - 1:
            return nums[idx]
        if idx < k - 1:
            left = idx + 1
        else:
            right = idx - 1
    return -1

def partition(first_index, last_index, nums):
    pivot = (first_index + last_index) // 2
    nums[pivot], nums[last_index] = nums[last_index], nums[pivot]

    for i in range(first_index, last_index):
        if nums[i] >= nums[last_index]:
            nums[i], nums[first_index] = nums[first_index], nums[i]
            first_index += 1
    nums[first_index], nums[last_index] = nums[last_index], nums[first_index]
    return first_index

# nums = [3,2,3,1,2,4,5,5,6]
# k = 4
# print(kth_largest(nums, k))

##278. First Bad Version
def firstBadVersion(n):
    left = 1
    right = n
    while left <= right:
        mid = left + (right-left)//2
        if isBad(mid):
            result = mid
            right = mid - 1
        else:
            left = mid + 1
    return result


##102. Binary Tree Level Order Traversal
def BTLevel(root):
    res = []
    if root is None:
        return res
    stack = [root]
    res = [[root.val]]
    while len(stack) != 0:
        temp = []
        tempVal = []
        for cur in stack:
            if cur.left:
                temp.append(cur.left)
                tempVal.append(cur.left.val)
            if cur.right:
                temp.append(cur.right)
                tempVal.append(cur.right.val)

        if len(tempVal) != 0:
            res.append(tempVal)
        stack = temp
    return res

##341. Flatten Nested List Iterator
class flattenNextedList:
    def __init__(self, nestedList):
        self.nestedList = nestedList
        self.res = []
        self.nested(self.nestedList, self.res)
        self.pointer = -1


    def nested(self, list, res):
        if not list:
            return
        for i in list:
            if i.isInteger():
                res.append(i.getInteger())
            else:
                self.nested(i.getList())

    def next(self):
        self.pointer += 1
        return self.res[self.pointer]

    def hasNext(self):
        return self.pointer < len(self.res) - 1

##75. Sort Colors
# Input: nums = [2,0,2,1,1,0]
# Output: [0,0,1,1,2,2]

def sortColors(nums):
    left = 0
    current = 0
    right = len(nums) - 1

    while current <= right:
        if nums[current] == 0:
            nums[current], nums[left] = nums[left], nums[current]
            current += 1
            left += 1

        elif nums[current] == 2:
            nums[right], nums[current] = nums[current], nums[right]
            right -= 1
        else:
            current += 1
    return nums
# nums = [2, 0, 1]
# print(sortColors(nums))

##647. Palindromic Substrings
# Input: s = "abc"
# Output: 3
# Explanation: Three palindromic strings: "a", "b", "c".

def palindromicSubstring(s):
    total = 0
    for i in range(len(s)):
        total += palCount(i, i, s)
        total += palCount(i, i+1, s)
    return total

def palCount(left, right, s):
    count = 0
    while left >= 0 and right < len(s) and s[left] == s[right]:
        count += 1
        left -= 1
        right += 1
    return count

##Time complexity O(n^2) space O(1)
# s = "aaa"
# print(palindromicSubstring(s))

##150. Evaluate Reverse Polish Notation
# Input: tokens = ["2","1","+","3","*"]
# Output: 9
# Explanation: ((2 + 1) * 3) = 9

def evalReversePolish(tokens):
    stack = []
    for i in tokens:
        if i in "+-*/":
            first = stack.pop()
            second = stack.pop()
            if i == "+":
                val = first + second
            elif i == "-":
                val = second - first
            elif i == "*":
                val = second*first
            else:
                val = int(second / first)
            stack.append(val)
        else:
            stack.append(int(i))
    return stack.pop()

# tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
# print(evalReversePolish(tokens))

##224. Basic Calculator
# Input: s = "(1+(4+5+2)-3)+(6+8)"
# Output: 23

def basicCalculator(s):
    stack = []
    sign = 1
    number = 0
    res = 0

    for c in s:
        if c.isdigit():
            number = number*10 + int(c)
        elif c == "+":
            res += number*sign
            sign = 1
            number = 0
        elif c == "-":
            res += number*sign
            sign = -1
            number = 0
        elif c == "(":
            res += number*sign
            stack.append(res)
            stack.append(sign)
            res = 0
            sign = 1
            number = 0
        elif c == ")":
            res += number*sign
            res *= stack.pop()
            res += stack.pop()

            sign = 1
            number = 0
    return res + number*sign
#
# Time O(n) space O(n)
# s = "1 + 1"
# print(basicCalculator(s))



##133. Clone Graph
class graphNode:
    def __init__(self, val, neighbours=None):
        self.val = val
        self.neighbours = neighbours if neighbours is not None else []

def cloneGraph(node):
    clone = {}
    clone[node] = graphNode(node.val, [])

    import collections
    queue = collections.deque([])
    queue.append(node)

    while queue:
        cur = queue.popleft()
        for nei in cur.neighbours:
            if nei not in clone:
                clone[nei] = graphNode(cur.val, [])
                queue.append(cur)
            clone[cur].neighbours.append(clone[nei])
    return clone[node]

##Time O(VE)  space O(n)


##253. Meeting Rooms II

def meetingRoom2(intervals):
    intervals.sort(key=lambda x:x[0])
    room = 0
    import heapq
    minHeap = []
    for i in range(len(intervals)):
        start = intervals[i][0]
        end = intervals[i][1]
        if len(minHeap) != 0 and start >= minHeap[0]:
            heapq.heappop(minHeap)
        else:
            room += 1
        heapq.heappush(minHeap, end)
    return room

# intervals = [[7,10],[2,4], [1,5],[10,30],[11,26],[30,40]]
# print(meetingRoom2(intervals))
# Time O(NlogN) space O(N)


##334. Increasing Triplet Subsequence

def increasingTriplet(nums):
    i = float('inf')
    j = float('inf')

    for k in range(len(nums)):
        if nums[k] <= i:
            i = nums[k]
        elif nums[k] <= j:
            j = nums[k]
        else:
            return True
    return False

##Time O(n) space O(1)
# nums = [2,1,5,0,4,6]
# print(increasingTriplet(nums))

# def maxTwo(nums):
#     max1 = float('-inf')
#     max2 = float('-inf')
#
#     for num in nums:
#         if num > max1:
#             max1, max2 = num, max1
#         elif max1 > num > max2:
#             max2 = num
#
#     return max1 , max2
#
# nums = [11,10,190,0,11,9]
# print(maxTwo(nums))

##209. Minimum Size Subarray Sum
# Input: target = 7, nums = [2,3,1,2,4,3]
# Output: 2
# Explanation: The subarray [4,3] has the minimal length under the problem constraint.

def minSizeSubArraySum(nums, target):
    cSum = 0
    res = len(nums) + 1
    left = 0
    for right in range(len(nums)):
        cSum += nums[right]
        while cSum >= target and left <= right:
            res = min(res, right-left+1)
            cSum -= nums[left]
            left += 1

    if res == len(nums) + 1:
        return 0
    else:
        return res

# nums = [2,3,1,2,4,3]
# target = 7
# print(minSizeSubArraySum(nums, target))

##252. Meeting Rooms
# Input: [[0,30],[5,10],[15,20]]
# Output: false

def meetingRoom(intervals):
    intervals.sort(key=lambda x:x[0])
    for i in range(1, len(intervals)):
        if intervals[i][0] < intervals[i-1][1]:
            return False
    return True

# intervals = [[7,10],[2,4]]
# print(meetingRoom(intervals))


##653. Two Sum IV - Input is a BST

def twoSumBST4(root, k):
    if root is None:
        return False
    nums = []
    stack = []

    while stack or root:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        nums.append(root.val)
        root = root.right

    left = 0
    right = len(nums) - 1

    while left <= right:
        if nums[left] + nums[right] == k:
            return True
        if nums[left] + nums[right] < k:
            left += 1
        else:
            right -= 1

    return False

##Time O(n) space O(n)
def invalidCount(s):
    stack = []
    for i in range(len(s)):
        if s[i] == "(":
            stack.append((i,s[i]))
        elif s[i] == ")":
            if len(stack) != 0 and stack[-1][1] == "(":
                stack.pop()
            else:
                stack.append((i,s[i]))
    sList = list(s)
    idxSet = set()
    for st in stack:
        idxSet.add(st[0])
    res = ""
    for i in range(len(sList)):
        if i not in idxSet:
            res += sList[i]
    return "".join(sList)

# s = "lee(t(c)o)de)"
# print(invalidCount(s))


##764. Largest Plus Sign
def largestPlusSign(n, mines):
    table = [[0]*n for _ in range(n)]
    banned = {tuple(mine) for mine in mines}
    ans = 0
    for i in range(n):
        count = 0
        for j in range(n):
            if (i,j) in banned:
                count = 0
            else:
                count += 1
            table[i][j] = count

        count = 0
        for j in range(n-1, -1, -1):
            if (i,j) in banned:
                count = 0
            else:
                count += 1
            if table[i][j] > count:
                table[i][j] = count

    for j in range(n):
        count = 0
        for i in range(n):
            if (i,j) in banned:
                count = 0
            else:
                count += 1
            if table[i][j] > count:
                table[i][j] = count

        count = 0
        for i in range(n-1, -1, -1):
            if (i,j) in banned:
                count = 0
            else:
                count += 1
            if table[i][j] > count:
                table[i][j] = count
            if table[i][j] > ans:
                ans = table[i][j]
    return ans

# n = 5
# mines = [[4,2]]
# print(largestPlusSign(n, mines))
##Time O(n^2)  space O(n^2)


##784. Letter Case Permutation
def letterCasePermutation(s):
    import collections
    queue = collections.deque([s])
    for i in range(len(s)):
        if s[i].isalpha():
            size = len(queue)
            while size != 0:
                curStr = queue.popleft()
                left = curStr[:i] + curStr[i].lower() + curStr[i+1:]
                right = curStr[:i] + curStr[i].upper() + curStr[i+1:]
                queue.append(left)
                queue.append(right)
                size -= 1
    return list(queue)

# s = "a1b2"
# print(letterCasePermutation(s))
##Time O(2^n) n -- number of letter in string space O(n*2^n)

##785. Is Graph Bipartite?

def isBipartite(graph):
    ##using Union find
    ##node and its neighbour should have different parent
    parent = {}
    rank = {}
    for node in range(len(graph)):
        parent[node] = node
        rank[node] = 0
    for node, neighbours in enumerate(graph):
        for nei in neighbours:
            if find(parent, node) == find(parent, nei):
                return False
            union(parent, rank, nei, neighbours[0])
    return True

def find(parent, x):
    if parent[x] != x:
        parent[x] = find(parent, parent[x])
    return parent[x]

def union(parent, rank, x, y):
    x_parent = find(parent, x)
    y_parent = find(parent, y)
    if x_parent != y_parent:
        if rank[x_parent] > rank[y_parent]:
            parent[y_parent] = x_parent
        elif rank[y_parent] > rank[x_parent]:
            parent[x_parent] = y_parent
        else:
            parent[y_parent] = x_parent
            rank[x_parent] += 1

# graph = [[1,3],[0,2],[1,3],[0,2]]
# print(isBipartite(graph))
##Time O(ElogV) space O(V)


##794. Valid Tic-Tac-Toe State
def validTicTacToe(grid):
    xCount = 0
    oCount = 0
    for row in grid:
        for char in row:
            if char == "X":
                xCount += 1
            if char == "O":
                oCount += 1

    def win(grid, player):
        for i in range(len(grid)):
            if grid[i][0] ==player and grid[i][1] == player and grid[i][2] == player:
                return True
        for j in range(len(grid[0])):
            if grid[0][j] == player and grid[1][j] == player and grid[2][j] == player:
                return True
        return (grid[0][0] == grid[1][1] == grid[2][2] == player) or (grid[0][2] == grid[1][1] == grid[2][0] == player)

    if oCount not in {xCount, xCount - 1}:
        return False
    if win(grid, "X") and xCount != oCount + 1:
        return False
    if win(grid, "O") and xCount != oCount:
        return False
    return True

# grid = ["XOX","O O","XOX"]
# print(validTicTacToe(grid))
