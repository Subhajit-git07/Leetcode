# 1. Two Sum
# Input: nums = [2,7,11,15], target = 9
# Output: [0,1]
# Output: Because nums[0] + nums[1] == 9, we return [0, 1].

def twoSum(nums, target):
    numDict = {}
    for i in range(len(nums)):
        if target-nums[i] in numDict:
            return [numDict[target-nums[i]], i]

        else:
            numDict[nums[i]] = i

# nums = [2,7,11,15]
# target = 9
# print(twoSum(nums, target))


##2. Add Two Numbers
class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None

def addTwoNums(l1, l2):
    cur = ListNode()
    head = cur
    if l1 is None:
         return l2
    if l2 is None:
        return l1

    carry = 0
    while l1 or l2 or carry:
        val1 = l1.val if l1 else 0
        val2 = l2.val if l2 else 0

        value = val1 + val2 + carry
        value = value % 10
        carry = value // 10

        cur.next = ListNode(value)
        cur = cur.next

        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None

    return head.next

##3. Longest Substring Without Repeating Characters

# Input: s = "abcabcbb"
# Output: 3
# Explanation: The answer is "abc", with the length of 3.

def longestStrNorepeat(s):
    seen = {}
    start = 0
    maxLen = 0
    res = ""

    for i in range(len(s)):
        if s[i] in seen:
            start = max(start, seen[s[i]]+1)
        seen[s[i]] = i
        maxLen = max(maxLen, i-start+1)
        if len(res) < len(s[start:i+1]):
            res = s[start:i+1]
    return res
#
# s = "abcabcbb"
# print(longestStrNorepeat(s))

class TrieNode:
    def __init__(self):
        self.children = {}
        self.isEnd = False

class trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        parent = self.root
        for i, char in enumerate(word):
            if char not in parent.children:
                parent.children[char] = TrieNode()
            parent = parent.children[char]
        parent.isEnd = True

    def search(self, word):
        parent = self.root
        for i, char in enumerate(word):
            if char not in parent.children:
                return False
            parent = parent.children[char]
        return parent.isEnd

    def startsWith(self, prefix):
        parent = self.root
        for i, char in enumerate(prefix):
            if char not in parent.children:
                return False
            parent = parent.children[char]
        return True

    def search2(self, word):
        parent = self.root
        cur = [parent]
        for i, char in enumerate(word):
            newCur = []
            for node in cur:
                for key, value in node.children.items():
                    if char == "." or char == key:
                        if i == len(word) - 1 and value.isEnd:
                            return True
                        newCur.append(value)
            cur = newCur
        return False


# t = trie()
# t.insert("apple")
# print(t.search("apple"))
# print(t.search("app "))
# print(t.startsWith("app"))
# t.insert("app")
# print(t.search2(".pp"))

def rearrangeStr(s):
    alpha = "abcdefghijklmnopqrstuvwxyz"
    digits = "0123456789"
    special = "!@#$%^&*"

    left = 0
    right = len(s) - 1
    current = 0
    s = list(s)
    print(s)
    alphaStr = ""
    nums = ""
    specialStr = ""

    for c in s:
        if c in alpha:
            alphaStr += c
        elif c in digits:
            nums += c
        else:
            specialStr += c
    return alphaStr + nums + specialStr


    # while current <= right:
    #     if s[current] in alpha:
    #         s[current], s[left] = s[left], s[current]
    #         left += 1
    #         current += 1
    #     elif s[current] in special:
    #         s[current], s[right] = s[right], s[current]
    #         right -= 1
    #     else:
    #         current += 1
    print(s)
    return "".join(s)

# s = "ab23^&ww08h"
# print(rearrangeStr(s))

##719. Find K-th Smallest Pair Distance
# Input: nums = [1,3,1], k = 1
# Output: 0
# Explanation: Here are all the pairs:
# (1,3) -> 2
# (1,1) -> 0
# (3,1) -> 2
# Then the 1st smallest distance pair is (1,1), and its distance is 0.

def kThsmallestDisPair(nums, k):
    nums.sort()
    left = 0
    right = nums[-1] - nums[0]
    while left <= right:
        mid = left + (right-left)//2
        count = condition(nums, k, mid)
        if count >= k:
            right = mid - 1
        else:
            left = mid + 1
    return left

def condition(nums, k, mid):
    count = 0
    left = 0
    for right in range(1, len(nums)):
        if nums[right] - nums[left] > mid:
            left += 1
        count += right - left
    return count

# nums = [1,3,1]
# k = 1
# print(kThsmallestDisPair(nums, k))

##676. Implement Magic Dictionary
class magicDict:
    def __init__(self):
        self.dict1 = {}

    def build(self, wordLists):
        for word in wordLists:
            if len(word) in self.dict1:
                self.dict1[len(word)].append(word)
            else:
                self.dict1[len(word)] = [word]

    def search(self, searchWord):
        if len(searchWord) not in self.dict1:
            return False
        words = self.dict1[len(searchWord)]
        for word in words:
            count = 0
            for i in range(len(searchWord)):
                if word[i] != searchWord[i]:
                    count += 1
                if count > 1:
                    break
            if count == 1:
                return True
        return False

##680. Valid Palindrome II

# Input: s = "abca"
# Output: true
# Explanation: You could delete the character 'c'.

def validPelindrome2(s):
    left = 0
    right = len(s) - 1

    while left <= right:
        if s[left] != s[right]:
            return isPalin(s, left+1, right) or isPalin(s, left, right-1)
        left += 1
        right -= 1
    return True

def isPalin(s, left, right):
    while left <= right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True

# s = "abca"
# print(validPelindrome2(s))

# print("i"<"love")

##33. Search in Rotated Sorted Array
# Input: nums = [4,5,6,7,0,1,2], target = 0
# Output: 4

def search(nums, target):
    if len(nums) == 0:
        return -1
    left = 0
    right = len(nums) - 1

    while left <= right:
        mid = left + (right-left)//2
        if nums[mid] == target:
            return mid
        elif nums[mid] >= nums[left]:
            if target >= nums[left] and target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if target > nums[mid] and target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1
# nums = [4,5,6,7,0,1,2]
# target = 0
# print(search(nums, target))

#34. Find First and Last Position of Element in Sorted Array
# Input: nums = [5,7,7,8,8,10], target = 8
# Output: [3,4]

def searchRange(nums, target):
    left = searchRange_helper(nums, target, True)
    right = searchRange_helper(nums, target, False)
    return [left, right]

def searchRange_helper(nums, target, leftBias):
    i = -1
    left = 0
    right = len(nums) - 1

    while left <= right:
        mid = left + (right-left)//2
        if target > nums[mid]:
            left = mid + 1
        elif target < nums[mid]:
            right = mid - 1
        else:
            i = mid
            if leftBias:
                right = mid - 1
            else:
                left = mid + 1
    return i

# nums = [5,7,7,8,8,10]
# target = 8
# print(searchRange(nums, target))

##35. Search Insert Position
# Input: nums = [1,3,5,6], target = 5
# Output: 2

def searchInsert(nums, target):
    if len(nums) == 0:
        return 0
    left = 0
    right = len(nums) - 1

    while left <= right:
        mid = left + (right-left)//2
        if nums[mid] == target:
            return mid
        if nums[mid] > target:
            right = mid - 1
        else:
            left = mid + 1
    return left

# nums = [1,3,5,6]
# target = 2
# print(searchInsert(nums, target))

##41. First Missing Positive
# Input: nums = [3,4,-1,1]
# Output: 2

def firstMissingPositive(nums):
    n = len(nums)
    for i in range(len(nums)):
        if nums[i] <= 0:
            nums[i] = n + 1

    for i in range(len(nums)):
        if abs(nums[i]) > n:
            continue

        pos = abs(nums[i]) - 1
        if nums[pos] > 0:
            nums[pos] = -1*nums[pos]
    for i in range(len(nums)):
        if nums[i] > 0:
            return i + 1
    return n + 1

# nums = [7,8,9,11,12]
# print(firstMissingPositive(nums))

##53. Maximum Subarray
# Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
# Output: 6
# Explanation: [4,-1,2,1] has the largest sum = 6.

def maxSubArray(nums):
    if len(nums) == 0:
        return 0
    maxByEnd = nums[0]
    maxSoFar = nums[0]

    for i in range(1, len(nums)):
        maxByEnd = max(nums[i], maxByEnd+nums[i])
        maxSoFar = max(maxSoFar, maxByEnd)
    return maxSoFar
# nums = [-2,1,-3,4,-1,2,1,-5,4]
# print(maxSubArray(nums))

##62. Unique Paths
def uniquePaths(m, n):
    if m == 0 or n == 0:
        return 0
    table = [[0 for col in range(n)] for row in range(m)]
    for i in range(m):
        table[i][0] = 1
    for j in range(n):
        table[0][j] = 1

    for i in range(1, m):
        for j in range(1, n):
            table[i][j] = table[i-1][j] + table[i][j-1]
    return table[m-1][n-1]

# print(uniquePaths(300,101))

##63. Unique Paths II
def uniquePaths2(grid):
    if grid[0][0] == 1:
        return 0
    grid[0][0] = 1
    m = len(grid)
    n = len(grid[0])

    for i in range(1, m):
        if grid[i][0] == 0:
            grid[i][0] += grid[i-1][0]
        else:
            grid[i][0] = 0
    for j in range(1, n):
        if grid[0][j] == 0:
            grid[0][j] += grid[0][j-1]
        else:
            grid[0][j] = 0

    for i in range(1, m):
        for j in range(1, n):
            if grid[i][j] == 0:
                grid[i][j] = grid[i-1][j] + grid[i][j-1]
            else:
                grid[i][j] = 0
    return grid[m-1][n-1]

# grid = [[0,1],[0,0]]
# print(uniquePaths2(grid))


##64. Minimum Path Sum
def minPathSum(grid):
    m = len(grid)
    n = len(grid[0])

    for i in range(1, m):
        grid[i][0] += grid[i-1][0]
    for j in range(1, n):
        grid[0][j] += grid[0][j-1]

    for i in range(1, m):
        for j in range(1, n):
            grid[i][j] += min(grid[i-1][j], grid[i][j-1])
    return grid[m-1][n-1]

# grid = [[1,2,3],[4,5,6]]
# print(minPathSum(grid))

##70. Climbing Stairs
def climbStaris(n):
    table = [0]*(n+1)
    if n == 1:
        return 1
    if n == 2:
        return 2
    table[1] = 1
    table[2] = 2
    for i in range(3, n+1):
        table[i] = table[i-1] + table[i-2]
    return table[n]

# print(climbStaris(5))

##695. Max Area of Island
def maxAreaOfIsland(grid):
    if len(grid) == 0 or len(grid[0]) == 0:
        return 0
    maxArea = 0
    visited = [[False for col in range(len(grid[0]))] for row in range(len(grid))]
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                maxArea = max(maxArea, maxAreaOfIsland_dfs(grid, i, j, visited))
    return maxArea

def maxAreaOfIsland_dfs(grid, i, j, visited):
    if i < 0 or i >= len(grid) or j <0 or j >= len(grid[0]) or visited[i][j]:
        return 0
    if grid[i][j] == 0:
        return 0
    visited[i][j] = True
    count = 1
    count += maxAreaOfIsland_dfs(grid, i+1, j, visited)
    count += maxAreaOfIsland_dfs(grid, i-1, j, visited)
    count += maxAreaOfIsland_dfs(grid, i, j+1, visited)
    count += maxAreaOfIsland_dfs(grid, i, j-1, visited)
    return count

# grid = [[0,0,0,0,0,0,0,0]]
#
# print(maxAreaOfIsland(grid))

#697. Degree of an Array

def degreeOfArray(nums):
    #store the left most index
    leftDict = {}
    ##Store the right most index
    rightDict = {}
    counter = {}

    for i in range(len(nums)):
        if nums[i] not in leftDict:
            leftDict[nums[i]] = i
        rightDict[nums[i]] = i
        counter[nums[i]] = counter.get(nums[i], 0) + 1

    degree = max(counter.values())
    ans = len(nums)
    for key in counter:
        if counter[key] == degree:
            ans = min(ans, rightDict[key] - leftDict[key] + 1)
    return ans

# nums = [1,2,2,3,1,4,2]
# print(degreeOfArray(nums))

##700. Search in a Binary Search Tree
def searchInBST(root, val):
    if root is None:
        return None
    if root.val == val:
        return root
    if val < root.val:
        searchInBST(root.left, val)
    else:
        searchInsert(root.right, val)

##701. Insert into a Binary Search Tree
def insertInBST(root, val):
    if root is None:
        root = TreeNode(val)
        return root
    insert_(root, val)
    return root

def insert_(root, val):
    if val < root.val:
        if root.left:
            insert_(root.left, val)
        else:
            root.left = TreeNode(val)

    if val > root.right:
        if root.right:
            insert_(root.right, val)
        else:
            root.right = TreeNode(val)


##703. Kth Largest Element in a Stream
import heapq
class kThLargest:

    def __init__(self,k, nums):

        self.k = k
        self.heap = nums
        heapq.heapify(self.heap)
        while len(self.heap) > k:
            heapq.heappop(self.heap)

    def add(self, val):
        heapq.heappush(self.heap, val)
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)
        return self.heap[0]

# kl = kThLargest(3, [4, 5, 8, 2])
# print(kl.add(3)) # return 4
# print(kl.add(5))   #return 5
# print(kl.add(10))  #return 5
# print(kl.add(9))  # return 8
# print(kl.add(4))  # return 8

##733. Flood Fill

# Input: image = [[1,1,1],[1,1,0],[1,0,1]], sr = 1, sc = 1, newColor = 2
# Output: [[2,2,2],[2,2,0],[2,0,1]]

def floodFill(grid, sr, sc, newColor):
    prevColor = grid[sr][sc]
    visited = [[False for col in range(len(grid[0]))] for row in range(len(grid))]
    floodFill_BFS(grid, sr, sc, prevColor, newColor, visited)
    return grid



def floodFill_BFS(grid, i, j, prevColor, newColor, visited):
    if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]):
        return
    if visited[i][j]:
        return
    if grid[i][j] != prevColor:
        return
    visited[i][j] = True
    grid[i][j] = newColor
    floodFill_BFS(grid, i+1, j, prevColor, newColor, visited)
    floodFill_BFS(grid, i-1, j, prevColor, newColor, visited)
    floodFill_BFS(grid, i, j+1, prevColor, newColor, visited)
    floodFill_BFS(grid, i, j-1, prevColor, newColor, visited)

# grid = [[0,0,0],[0,0,0]]
# sr = 0
# sc = 0
# newColor = 2
# print(floodFill(grid, sr, sc, newColor))


##740. Delete and Earn
# Input: nums = [3,4,2]
# Output: 6

def deleteAndEarn(nums):
    Max = max(nums)
    if Max == 1:
        return len(nums)
    if len(nums) == 0:
        return 0
    table = [0]*(Max+1)
    for num in nums:
        table[num] += num
    n = len(table)
    table[n-2] = max(table[n-2], table[n-1])
    for i in range(n-3, -1, -1):
        table[i] = max(table[i+1], table[i]+table[i+2])
    return table[0]

# nums = [3,4,2]
# print(deleteAndEarn(nums))

##Prefix and Suffix Search
class Node:
    def __init__(self):
        self.children = {}
        self.maxIdx = 0

class prefixSuffixSearch:
    def __init__(self, words):
        self.root = Node()
        #self.words = words

        for i, word in enumerate(words):
            nWord = word + "#" + word
            for j in range(len(word)):
                parent = self.root
                for k in range(j, len(nWord)):
                    char = nWord[k]
                    if char not in parent.children:
                        parent.children[char] = Node()
                    parent = parent.children[char]
                    parent.maxIdx = i

    def search(self, prefix, suffix):
        searchWord = suffix + "#" + prefix
        parent = self.root
        for i, char in enumerate(searchWord):
            if char not in parent.children:
                return -1
            parent = parent.children[char]
        return parent.maxIdx

# preSuf = prefixSuffixSearch(["apple", "ball", "food"])
# print(preSuf.search("app", "e"))

##746. Min Cost Climbing Stairs
# Input: cost = [1,100,1,1,1,100,1,1,100,1]
# Output: 6

def mincostClimbing(cost):
    if len(cost) == 0:
        return 0
    if len(cost) == 1:
        return cost[0]
    table = [0]*len(cost)
    if len(cost) == 1:
        return cost[0]
    if len(cost) == 2:
        return min(cost[0], cost[1])
    table[0] = cost[0]
    table[1] = cost[1]
    n = len(table)
    for i in range(2, len(cost)):
        table[i] = min(cost[i]+table[i-2], cost[i]+table[i-1])
    return min(table[n-1], table[n-2])

# cost = [1,100,1,1,1,100,1,1,100,1]
# print(mincostClimbing(cost))


##752. Open the Lock
# Input: deadends = ["0201","0101","0102","1212","2002"], target = "0202"
# Output: 6

def openTheLock(deadends, target):
    #deadends = set(deadends)
    visited = set()
    level = 0
    for end in deadends:
        visited.add(end)

    stack = ['0000']
    while len(stack) != 0:
        tempLevel = []
        for cur in stack:
            if cur == target:
                return level
            if cur not in visited:
                visited.add(cur)
                cur = list(cur)
                for i in range(len(cur)):
                    temp = cur[i]
                    if int(cur[i]) == 9:
                        cur[i] = "0"
                        newWord1 = "".join(cur)
                        cur[i] = "8"
                        newWord2 = "".join(cur)
                    elif int(cur[i]) == 0:
                        cur[i] = "1"
                        newWord1 = "".join(cur)
                        cur[i] = "9"
                        newWord2 = "".join(cur)
                    else:
                        cur[i] = str(int(cur[i]) + 1)
                        newWord1 = "".join(cur)
                        cur[i] = str(int(cur[i]) - 1)
                        newWord2 = "".join(cur)
                    cur[i] = temp
                    if newWord1 not in visited:
                        tempLevel.append(newWord1)
                    if newWord2 not in visited:
                        tempLevel.append(newWord2)
        level += 1
        print(tempLevel)
        stack = tempLevel
    return -1

# deadends = ["0201","0101","0102","1212","2002"]
# target = "0202"
# print(openTheLock(deadends, target))

##763. Partition Labels
# Input: s = "ababcbacadefegdehijhklij"
# Output: [9,7,8]

def partitionLevels(s):
    dictS = {}
    for i in range(len(s)):
        dictS[s[i]] = i
    start = 0
    res = []
    output = []
    end = 0

    for i in range(len(s)):
        end = max(end, dictS[s[i]])
        if i == end:
            res.append(i-start+1)
            output.append(s[start:i+1])
            start = i + 1
    return output

# s = "eccbbbbdec"
# print(partitionLevels(s))

##764. Largest Plus Sign
# Input: n = 5, mines = [[4,2]]
# Output: 2

def largestPlusSign(n, mines):
    banned = {tuple(mine) for mine in mines}
    res = 0
    table = [[0 for col in range(n)] for row in range(n)]

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
            if count < table[i][j]:
                table[i][j] = count

    for j in range(n):
        count = 0
        for i in range(n):
            if (i,j) in banned:
                count = 0
            else:
                count += 1
            if count < table[i][j]:
                table[i][j] = count

        count = 0
        for i in range(n-1, -1, -1):
            if (i,j) in banned:
                count = 0
            else:
                count += 1
            if count < table[i][j]:
                table[i][j] = count
            res = max(res, table[i][j])
    return res

# n = 1
# mines = [[0,0]]
# print(largestPlusSign(n, mines))

##221. Maximal Square
def maximalSquare(grid):
    m = len(grid)
    n = len(grid[0])
    height = 0

    table = [[0 for col in range(n+1)] for row in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if grid[i-1][j-1] == "1":
                table[i][j] = 1 + min(table[i-1][j], table[i][j-1], table[i-1][j-1])
                height = max(height, table[i][j])
    return height*height

# grid = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
# print(maximalSquare(grid))

##765. Couples Holding Hands
# Input: row = [0,2,1,3]  0 1 2 3
# Output: 1

def coupleHoldHands(row):
    rowDict = {}
    for i in range(len(row)):
        rowDict[row[i]] = i

    swaps = 0
    for i,val in enumerate(row):
        if i%2 == 0:
            if val%2 == 0:
                if row[i+1] != val + 1:
                    tempIdx = rowDict[val + 1]
                    row[i+1], row[tempIdx] = row[tempIdx], row[i+1]
                    rowDict[row[i+1]] = i+1
                    rowDict[row[tempIdx]] = tempIdx
                    swaps += 1
            else:
                if row[i+1] != val - 1:
                    tempIdx = rowDict[val-1]
                    row[i+1], row[tempIdx] = row[tempIdx], row[i+1]
                    rowDict[row[i+1]] = i+1
                    rowDict[row[tempIdx]] = tempIdx
                    swaps += 1
    return swaps

# row = [3,2,0,1]
# print(coupleHoldHands(row))

##767. Reorganize String
# Input: s = "aab"
# Output: "aba"

def reorganiseStr(s):
    if len(s) == 1:
        return s
    strDict = {}
    for c in s:
        if c in strDict:
            strDict[c] += 1
        else:
            strDict[c] = 1
    import heapq
    heap = []
    for key in strDict:
        heapq.heappush(heap, (-strDict[key], key))
    res = ""

    while len(heap) > 1:
        first = heapq.heappop(heap)
        second = heapq.heappop(heap)
        res += first[1]
        res += second[1]

        if first[0] + 1 != 0:
            heapq.heappush(heap, (first[0]+1, first[1]))
        if second[0] + 1 != 0:
            heapq.heappush(heap, (second[0]+1, second[1]))

    if len(heap) != 0:
        last = heapq.heappop(heap)
        if last[0] + 1 != 0:
            return ""
        res += last[1]
    return res

# s = "aaab"
# print(reorganiseStr(s))


##784. Letter Case Permutation
# Input: s = "a1b2"
# Output: ["a1b2","a1B2","A1b2","A1B2"]

def letterCasePermu(s):
    from collections import deque
    queue = deque()
    queue.append(s)

    for i in range(len(s)):
        if s[i].isalpha():
            size = len(queue)
            while size != 0:
                cur = queue.popleft()
                left = cur[:i] + cur[i].upper() + cur[i+1:]
                right = cur[:i] + cur[i].lower() + cur[i+1:]
                queue.append(left)
                queue.append(right)
                size -= 1
    return queue

# s = "a1b2"
# print(letterCasePermu(s))

##785. Is Graph Bipartite?
def bipartite(graph):
    color = [0]*len(graph)
    from collections import deque
    #queue = deque()

    for i in range(len(graph)):
        if color[i] == 0:
            queue = deque([])
            queue.append(i)
            color[i] = 1
            while len(queue) != 0:
                node = queue.popleft()
                for neighbour in graph[node]:
                    if color[neighbour] == color[node]:
                        return False
                    elif color[neighbour] == 0:
                        queue.append(neighbour)
                        color[neighbour] = -color[node]
    return True

# graph = [[1,3],[0,2],[1,3],[0,2]]
# print(bipartite(graph))
# Time O(V+E) space O(V)

##794. Valid Tic-Tac-Toe State
def validTicTacState(grid):
    xCount = 0
    oCount = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == "X":
                xCount += 1
            elif grid[i][j] == "O":
                oCount += 1
    def win(grid, player):
        for i in range(len(grid)):
            if player == grid[i][0] == grid[i][1] == grid[i][2]:
                return True
        for j in range(len(grid[0])):
            if player == grid[0][j] == grid[1][j] == grid[2][j]:
                return True
        return player == grid[0][0] == grid[1][1] == grid[2][2] or player == grid[0][2] == grid[1][1] == grid[2][0]

    if oCount not in (xCount, xCount-1):
        return False
    if win(grid, "X") and xCount != oCount + 1:
        return False
    if win(grid, "O") and xCount != oCount:
        return False
    return True

# grid = ["XOX","O O","XOX"]
# print(validTicTacState(grid))

##875. Koko Eating Bananas

# Input: piles = [3,6,7,11], h = 8
# Output: 4

def minEatingSpeed(piles, h):
    left = 1
    right = max(piles)
    while left <= right:
        mid = left + (right-left)//2
        if speedHelper(piles, h, mid):
            right = mid - 1
        else:
            left = mid + 1
    return left


def speedHelper(piles, h, mid):
    totalHr = 0
    for pile in piles:
        totalHr += pile // mid
        if pile%mid != 0:
            totalHr += 1
    return totalHr <= h

# piles = [30,11,23,4,20]
# h = 6
# print(minEatingSpeed(piles, h))

##876. Middle of the Linked List

def middleOfList(head):
    slow = head
    fast = head

    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next

    return slow

##901. Online Stock Span
class onlineStockSpan:
    def __init__(self):
        self.res = 0
        self.idx = -1
        self.stack = []

    def next(self, price):
        self.idx += 1
        while self.stack and price >= self.stack[-1][1]:
            self.stack.pop()
        if self.stack:
            self.res = self.idx - self.stack[-1][0]
        else:
            self.res = self.idx + 1

        self.stack.append((self.idx, price))
        return self.res

# span = onlineStockSpan()
# print(span.next(100))
# print(span.next(80))
# print(span.next(60))
# print(span.next(70))
# print(span.next(60))
# print(span.next(75))
# print(span.next(85))

##980. Unique Paths III
def uniquePath3(grid):
    zero = 0
    startX = 0
    startY = 0

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 0:
                zero += 1
            elif grid[i][j] == 1:
                startX = i
                startY = j
    return uniquePath3_DFS(grid, startX, startY, zero)

def uniquePath3_DFS(grid, i, j, zero):
    if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] == -1:
        return 0
    if grid[i][j] == 2:
        if zero == -1:
            return 1
        else:
            return 0
    grid[i][j] = -1
    zero -= 1
    paths = 0
    paths += uniquePath3_DFS(grid, i+1, j, zero)
    paths += uniquePath3_DFS(grid, i-1, j, zero)
    paths += uniquePath3_DFS(grid, i, j+1, zero)
    paths += uniquePath3_DFS(grid, i, j-1, zero)

    grid[i][j] = 0  ##back tracking
    zero += 1

    return paths

# grid = [[0,1],[2,0]]
# print(uniquePath3(grid))

##993. Cousins in Binary Tree
def cousin(root, x, y):
    if root is None:
        return None
    if root.val == x or root.val == y:
        return False
    xList = cousin_helper(root, None, x, 0, info)
    yList = cousin_helper(root, None, y, 0, info)
    return xList[0][0] != yList[0][0] and xList[0][1] == yList[0][1]


def cousin_helper(root, parent, cousin, depth, info):
    if root is None:
        return None
    if root.val == cousin:
        info.append((parent.val, depth))
    cousin_helper(root.left, root, cousin, depth+1, info)
    cousin_helper(root.right, root, cousin, depth + 1, info)
    return info


##1002. Find Common Characters
# Input: words = ["bella","label","roller"]
# Output: ["e","l","l"]

def commonChar(words):
    charDict = {}
    for char in words[0]:
        if char in charDict:
            charDict[char] += 1
        else:
            charDict[char] = 1

    for i in range(1, len(words)):
        for key in charDict:
            charDict[key] = min(charDict[key], words[i].count(key))
    res = []
    for char in charDict:
        res += [char]*charDict[char]
    return res

# words = ["bella","label","roller"]
# print(commonChar(words))

##349. Intersection of Two Arrays
# Input: nums1 = [1,2,2,1], nums2 = [2,2]
# Output: [2]

def intersect(nums1, nums2):
    numSet1 = set(nums1)
    numSet2 = set(nums2)

    res = []
    for num in numSet1:
        if num in numSet2:
            res.append(num)
    return res

# nums1 = [4,9,5]
# nums2 = [9,4,9,8,4]
# print(intersect(nums1, nums2))

##350. Intersection of Two Arrays II
# Input: nums1 = [4,9,5], nums2 = [9,4,9,8,4]
# Output: [4,9]
# Explanation: [9,4] is also accepted.

def intersect2(nums1, nums2):
    numDict1 = {}
    numDict2 = {}

    for num in nums1:
        if num in numDict1:
            numDict1[num] += 1
        else:
            numDict1[num] = 1

    for num in nums2:
        if num in numDict2:
            numDict2[num] += 1
        else:
            numDict2[num] = 1

    res = []

    for num in numDict1:
        if num in numDict2:
            Min = min(numDict1[num], numDict2[num])
            res += [num]*Min

    return res

# nums1 = [1,2,2,1]
# nums2 = [2,2]
# print(intersect2(nums1, nums2))

#1008. Construct Binary Search Tree from Preorder Traversal
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def bstFromPreorder(preorder):
    root = None
    stack = []
    for num in preorder:
        if root is None:
            root = TreeNode(num)
            stack.append(root)
        else:
            node = TreeNode(num)
            if stack[-1].val > num:
                stack[-1].left = node
            else:
                while stack and stack[1].val < num:
                    u = stack.pop()
                u.right = node
            stack.append(node)
    return root

##1013. Partition Array Into Three Parts With Equal Sum
# Input: arr = [0,2,1,-6,6,-7,9,1,2,0,1]
# Output: true
# Explanation: 0 + 2 + 1 = -6 + 6 - 7 + 9 + 1 = 2 + 0 + 1

def threePartsWithEqualPartition(nums):
    Sum = sum(nums)
    if Sum%3 != 0:
        return False
    ideal = Sum // 3
    partition = 0
    temp = 0

    for num in nums:
        temp += num
        if temp == ideal:
            partition += 1
            temp = 0
    if partition >= 3:
        return True
    return False

# nums = [3,3,6,5,-2,2,5,1,-9,4]
# print(threePartsWithEqualPartition(nums))

##1991. Find the Middle Index in Array
# Input: nums = [2,3,-1,8,4]
# Output: 3
##total = 2left + nums[middle]

def middleIndex(nums):
    left = 0
    total = sum(nums)
    for i in range(len(nums)):
        if nums[i] == total - 2*left:
            return i
        left += nums[i]
    return -1

# nums = [1]
# print(middleIndex(nums))

##1035. Uncrossed Lines
# Input: nums1 = [1,4,2], nums2 = [1,2,4]
# Output: 2

def uncrossedLines(nums1, nums2):
    table = [[0 for col in range(len(nums2)+1)] for row in range(len(nums1)+1)]
    for i in range(1, len(table)):
        for j in range(1, len(table[0])):
            if nums1[i-1] == nums2[j-1]:
                table[i][j] = 1 + table[i-1][j-1]
            else:
                table[i][j] = max(table[i-1][j], table[i][j-1])
    return table[len(nums1)][len(nums2)]

#Time O(mn) space O(mn)
# nums1 = [1,4,2]
# nums2 = [1,2,4]
# print(uncrossedLines(nums1, nums2))

#1137. N-th Tribonacci Number
def nThTribonaci(n):
    table = [0]*(n+1)
    table[0] = 0
    table[1] = 1
    table[2] = 1

    for i in range(3, len(table)):
        table[i] = table[i-1] + table[i-2] + table[i-3]
    return table[n]

# n = 4
# print(nThTribonaci(25))

##1143. Longest Common Subsequence
# Input: text1 = "abcde", text2 = "ace"
# Output: 3
# Explanation: The longest common subsequence is "ace" and its length is 3.

#   "" a c e
# "" 0 0 0 0
# a 0  1 1 1
# b 0  1 1 1
# c 0  1 2 2
# d 0  1 2 2
# e 0  1 2 3

def LCS(text1, text2):
    if len(text1) == 0 or len(text2) == 0:
        return ""
    table = [["" for col in range(len(text2)+1)] for row in range(len(text1)+1)]
    for i in range(len(text1)+1):
        for j in range(len(text2)+1):
            if i == 0:
                table[0][j] = ""
            elif j == 0:
                table[i][0] = ""
            elif text1[i-1] == text2[j-1]:
                table[i][j] = table[i-1][j-1] + text1[i-1]
            else:
                if len(table[i-1][j]) >= len(table[i][j-1]):
                    table[i][j] = table[i-1][j]
                else:
                    table[i][j] = table[i][j-1]
    return table[len(text1)][len(text2)]

# text1 = "abc"
# text2 = "def"
# print(LCS(text1, text2))

##1628-Design-an-Expression-Tree-With-Evaluate-Function

class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def evalTree(root):
    if root is None:
        return 0
    if root.left is None and root.right is None:
        return root.val
    l = evalTree(root.left)
    r = evalTree(root.right)

    if root.val == "+":
        return l + r
    elif root.val == "-":
        return l - r
    elif root.val == "*":
        return l*r
    else:
        return l/r

