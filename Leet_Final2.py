##12. Integer to Roman

def intToRoman(num):
    res = ""
    symbolDict = {1000:"M", 900:"CM", 500:"D", 400:"CD", 100:"C", 90:"XC", 50:"L",
                  40:"XL", 10:"X", 9:"IX", 5:"V", 4:"IV", 1:"I"}
    for key in symbolDict:
        while num >= key:
            res += symbolDict[key]
            num -= key
    return res

# num = 1994
# print(intToRoman(num))
##Time O(length of result string) space O(1)

##13. Roman to Integer
def romanToInt(s):
    symbolDict = {"I":1, "V":5, "X":10, "L":50, "C":100, "D":500, "M":1000}

    res = symbolDict[s[len(s)-1]]
    for i in range(len(s)-2, -1, -1):
        if symbolDict[s[i]] < symbolDict[s[i+1]]:
            res -= symbolDict[s[i]]
        else:
            res += symbolDict[s[i]]

    return res

# s = "MCMXCIV"
# print(romanToInt(s))
##Time O(Len(s)) space O(1)


##16. 3Sum Closest
def threeSumClosest(nums, target):
    nums.sort()
    diff = float('inf')

    for i in range(len(nums)):
        left = i + 1
        right = len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if abs(target-total) < abs(diff):
                diff = target-total
            if total < target:
                left += 1
            else:
                right -= 1
    return target - diff

# nums = [-1,2,1,-4]
# target = 1
# print(threeSumClosest(nums, target))

##19. Remove Nth Node From End of List
def removeNthFromEnd(head, n):
    fast = head
    slow = head

    for i in range(n):
        fast = fast.next
    if fast is None:
        return head.next
    while fast.next != None:
        fast = fast.next
        slow = slow.next
    slow.next = slow.next.next
    return head
##Time O(n) space O(1)

##20. Valid Parentheses
def validParen(s):
    stack = []
    for i in range(len(s)):
        if s[i] in "({[":
            stack.append(s[i])
        else:
            if len(stack) == 0:
                return False
            else:
                first = stack.pop()
                if not isValid(first, s[i]):
                    return False
    if len(stack) == 0:
        return True
    else:
        return False

def isValid(p1, p2):
    if p1 == "(" and p2 == ")":
        return True
    elif p1 == "{" and p2 == "}":
        return True
    elif p1 == "[" and p2 == "]":
        return True
    else:
        return False

# s = "(]"
# print(validParen(s))

##21. Merge Two Sorted Lists
class ListNode:
    def __init__(self, val=None):
        self.val = val
        self.next = None

def mergeTwoSortedList(list1, list2):
    if list1 is None or list2 is None:
        return []
    temp = dummy = ListNode()
    while list1 and list2:
        if list1.val <= list2.val:
            temp = ListNode(list1.val)
            list1 = list1.next
        else:
            temp = ListNode(list2.val)
            list2 = list2.next
        temp = temp.next

    while list1:
        temp = ListNode(list1.val)
        list1 = list1.next
        temp = temp.next

    while list2:
        temp = ListNode(list2.val)
        list2 = list2.next
        temp = temp.next

    return dummy.next

##23. Merge k Sorted Lists
def mergeKSorted(lists):
    res = []
    if len(lists) == 0:
        return []
    if len(lists) == 1:
        return lists[0]
    total = len(lists)
    interval = 1

    while interval < total:
        for i in range(0, total-interval, interval*2):
            lists[i] = mergeTwoSortedList(lists[i], lists[i+interval])
        interval *= 2
    return lists[0]

##Time O(NlogK)  space O(N)  N = total number of nodes K = number of linked lists


##Sample
def divideConqureTest(lists):
    total = len(lists)
    interval = 1

    while interval < total:
        for i in range(0, total-interval, interval*2):
            lists[i] = lists[i] + lists[i+interval]
        print(lists)
        interval *= 2
    return lists[0]

# lists = [1, 6, 8, 4, 2]
# print(divideConqureTest(lists))


##57. Insert Interval
# Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
# Output: [[1,5],[6,9]]

def insertInterval(intervals, newInterval):
    res = []
    for i in range(len(intervals)):
        if intervals[i][1] < newInterval[0]:
            res.append(intervals[i])
        elif intervals[i][0] > newInterval[1]:
            res.append(newInterval)
            newInterval = intervals[i]
        else:
            newInterval[0] = min(intervals[i][0], newInterval[0])
            newInterval[1] = max(intervals[i][1], newInterval[1])

    res.append(newInterval)
    return res

# intervals = [[1,3],[6,9]]
# newInterval = [2,5]
# print(insertInterval(intervals, newInterval))
##Time O(N)  space O(N)


##56. Merge Intervals
# Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
# Output: [[1,6],[8,10],[15,18]]
# Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].

def mergeInterval(intervals):
    res = []
    inter = intervals[0]
    intervals.sort(key=lambda x:x[0])
    for i in range(1, len(intervals)):
        if inter[1] < intervals[i][0]:
            res.append(inter)
            inter = intervals[i]
        else:
            inter[0] = min(inter[0], intervals[i][0])
            inter[1] = max(inter[1], intervals[i][1])
    res.append(inter)
    return res

# intervals = [[1,4],[4,5]]
# print(mergeInterval(intervals))
##Time O(NlogN + N)  space O(N)

###54. Spiral Matrix
def spiralMatrix(matrix):
    top = 0
    down = len(matrix) - 1
    left = 0
    right = len(matrix[0]) - 1
    dir = 0
    res = []
    while top <= down and left <= right:
        if dir == 0:
            for i in range(left, right+1):
                res.append(matrix[top][i])
            top += 1

        elif dir == 1:
            for i in range(top, down+1):
                res.append(matrix[i][right])
            right -= 1

        elif dir == 2:
            for i in range(right, left-1, -1):
                res.append(matrix[down][i])
            down -= 1

        elif dir == 3:
            for i in range(down, top-1, -1):
                res.append(matrix[i][left])
            left += 1

        dir = (dir+1)%4

    return res

# matrix = [[1,2,3],[4,5,6],[7,8,9]]
# print(spiralMatrix(matrix))

##53. Maximum Subarray

def maxSubArray(nums):
    maxByEnd = nums[0]
    maxSoFar = nums[0]

    for i in range(1, len(nums)):
        maxByEnd = max(nums[i], maxByEnd+nums[i])
        maxSoFar = max(maxByEnd, maxSoFar)
    return maxSoFar

# nums = [-2,1,-3,4,-1,2,1,-5,4]
# print(maxSubArray(nums))
##Time O(n)  space O(1)


##50. Pow(x, n)

def myPow(x, n):
    res = 1
    isNegative = False
    if n < 0:
        inNegative = True
        n = n*(-1)

    while n > 0:
        if n%2 != 0:
            res = res*x
        n = n //2
        x = x*x

    if isNegative:
        return 1/res
    else:
        return res

# x = 2.10000
# n = 3
# print(myPow(x, n))
##Time O(logN)  space O(1)

##49. Group Anagrams
# Input: strs = ["eat","tea","tan","ate","nat","bat"]
# Output: [["bat"],["nat","tan"],["ate","eat","tea"]]

def groupAnagram(strs):
    dict1 = {}
    for word in strs:
        wordCount = [0]*26
        for c in word:
            pos = ord(c) - ord('a')
            wordCount[pos] += 1
        wordCount = tuple(wordCount)
        if wordCount in dict1:
            dict1[wordCount].append(word)
        else:
            dict1[wordCount] = [word]

    return list(dict1.values())

# strs = ["eat","tea","tan","ate","nat","bat"]
# print(groupAnagram(strs))
##Time O(NK) space O(N)

##48. Rotate Image
def rotateImage(grid):
    ##transpose the matrix
    for row in range(len(grid)):
        for col in range(row, len(grid[0])):
            grid[row][col], grid[col][row] = grid[col][row], grid[row][col]

    ##Reverse the rows
    for row in range(len(grid)):
        grid[row] = grid[row][::-1]

    return grid

# grid = [[1,2,3],[4,5,6],[7,8,9]]
# print(rotateImage(grid))

##43. Multiply Strings
# Input: num1 = "123", num2 = "456"
# Output: "56088"

def multiplyStr(nums1, nums2):
    if nums1 == '0' or nums2 == '0':
        return '0'
    res = 0
    for num1 in nums1:
        res = res*10 + multi(num1, nums2)
    return str(res)

def multi(num1, nums2):
    ans = 0
    for num2 in nums2:
        ans = ans*10 + (ord(num1)-ord('0'))*(ord(num2)-ord('0'))
    return ans

# nums1 = "0"
# nums2 = "3"
# print(multiplyStr(nums1, nums2))
##Time O(MN) space O(1)

##42. Trapping Rain Water
def tapWater(height):
    result = 0
    maxLeft = 0
    maxRight = 0
    left = 0
    right = len(height) - 1

    while left < right:
        maxLeft = max(maxLeft, height[left])
        maxRight = max(maxRight, height[right])
        if height[left] <= height[right]:
            result += maxLeft - height[left]
            left += 1
        else:
            result += maxRight - height[right]
            right -= 1
    return result

# height = [4,2,0,3,2,5]
# print(tapWater(height))
## Time O(N) space O(1)

##41. First Missing Positive
def firstMissingPositive(nums):
    n = len(nums)
    for i in range(n):
        if nums[i] <= 0:
            nums[i] = n+1

    for i in range(n):
        if abs(nums[i]) > n:
            continue
        pos = abs(nums[i]) - 1
        if nums[pos] > 0:
            nums[pos] = -nums[pos]

    for i in range(n):
        if nums[i] > 0:
            return i + 1
    return n + 1

# nums = [7,8,9,11,12]
# print(firstMissingPositive(nums))
##Time O(N) space O(1)


##36. Valid Sudoku
def validSudoku(grid):
    rowSet = set()
    colSet = set()
    blockSet = set()

    for i in range(9):
        for j in range(9):
            if grid[i][j] != ".":
                rKey = (i, grid[i][j])
                cKey = (j, grid[i][j])
                bKey = (i//3, j//3, grid[i][j])

                if rKey in rowSet or cKey in colSet or bKey in blockSet:
                    return False
                rowSet.add(rKey)
                colSet.add(cKey)
                blockSet.add(bKey)
    return True

# grid = [["8","3",".",".","7",".",".",".","."]
# ,["6",".",".","1","9","5",".",".","."]
# ,[".","9","8",".",".",".",".","6","."]
# ,["8",".",".",".","6",".",".",".","3"]
# ,["4",".",".","8",".","3",".",".","1"]
# ,["7",".",".",".","2",".",".",".","6"]
# ,[".","6",".",".",".",".","2","8","."]
# ,[".",".",".","4","1","9",".",".","5"]
# ,[".",".",".",".","8",".",".","7","9"]]
#
# print(validSudoku(grid))

##35. Search Insert Position
def insertPosition(nums, target):
    left = 0
    right = len(nums) - 1

    while left <= right:
        mid = left + (right-left)//2
        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return left

# nums = [1,3,5,6]
# target = 7
# print(insertPosition(nums, target))
##Time O(logN) space O(1)

##34. Find First and Last Position of Element in Sorted Array
def searchRange(nums, target):
    leftIndex = binarySearchRange(nums, target, True)
    rightIndex = binarySearchRange(nums, target, False)
    return [leftIndex, rightIndex]

def binarySearchRange(nums, target, leftBias):
    left = 0
    right = len(nums) - 1

    i = -1
    while left <= right:
        mid = left + (right-left)//2
        if nums[mid] < target:
            left = mid + 1
        elif nums[mid] > target:
            right = mid - 1
        else:
            i = mid
            if leftBias:
                right = mid - 1
            else:
                left = mid + 1
    return i

# nums = [5,7,7,8,8,10]
# target = 6
# print(searchRange(nums, target))
##Time O(LogN) space O(1)

##33. Search in Rotated Sorted Array
def searchRotated(nums, target):
    left = 0
    right = len(nums) - 1

    while left <= right:
        mid = left + (right-left)//2
        if nums[mid] == target:
            return mid
        if nums[left] <= nums[mid]:
            if nums[left] <= target and target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target and target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1

# nums = [4,5,6,7,0,1,2]
# target = 3
# print(searchRotated(nums, target))
##Time O(Logn)  space O(1)

###32. Longest Valid Parentheses
def longestValidParen(s):
    if len(s) == 0:
        return 0
    left = 0
    right = 0
    maxLen = 0
    start = 0
    end = 0
    res = ""
    for i in range(len(s)):
        if s[i] == "(":
            left += 1
        else:
            right += 1

        if left == right:
            maxLen = max(maxLen, left+right)
            Str = s[start:i+1]
            if len(Str) > len(res):
                res = Str
        if right > left:
            left = 0
            right = 0


    left = 0
    right = 0

    for i in range(len(s)-1, -1, -1):
        if s[i] == "(":
            left += 1
        else:
            right += 1

        if left == right:
            maxLen= max(maxLen, left+right)

        if left > right:
            left = 0
            right = 0

    return maxLen

# s = ")()())"
# print(longestValidParen(s))
##Time O(N)  space O(1)

##28. Implement strStr()
# Input: haystack = "hello", needle = "ll"
# Output: 2

def strStr(haystack, needle):
    if len(needle) == 0:
        return 0
    h = len(haystack)
    n = len(needle)
    if n > h:
        return -1
    for i in range(h-n+1):
        if haystack[i:i+n] == needle:
            return i
    return -1

# haystack = "hello"
# needle = "ll"
# print(strStr(haystack, needle))

##59. Spiral Matrix II
def spiralMatrix2(n):
    grid = [[0 for i in range(n)] for j in range(n)]
    top = 0
    down = len(grid) - 1
    left = 0
    right = len(grid[0]) - 1
    dir = 0
    val = 0

    while top <= down and left <= right:
        if dir == 0:
            for i in range(left, right+1):
                val += 1
                grid[top][i] = val
            top += 1
        elif dir == 1:
            for i in range(top, down+1):
                val += 1
                grid[i][right] = val
            right -= 1
        elif dir == 2:
            for i in range(right, left-1, -1):
                val += 1
                grid[down][i] = val
            down -= 1
        elif dir == 3:
            for i in range(down, top-1, -1):
                val += 1
                grid[i][left] = val
            left += 1
        dir = (dir+1)%4

    return grid

# n = 3
# print(spiralMatrix2(n))
# Time = O(n^2) space O(n^2)


##61. Rotate List
def rotateList(head, k):
    if head is None or k == 0:
        return head
    length = 0
    prev = None
    cur = head
    while cur:
        length += 1
        prev = cur
        cur = cur.next

    prev.next = head
    k = k%length

    rotate = length - k - 1
    cur = head
    while rotate > 0:
        cur = cur.next
        rotate -= 1
    head = cur.next
    cur.next = None
    return head

##Time O(n) space O(1)


##62. Unique Paths
def uniquePaths(m, n):
    grid = [[0 for i in range(n)] for j in range(m)]
    grid[0][0] = 1
    for i in range(m):
        grid[i][0] = 1
    for j in range(n):
        grid[0][j] = 1

    for i in range(1, m):
        for j in range(1, n):
            grid[i][j] = grid[i-1][j] + grid[i][j-1]
    return grid[m-1][n-1]

# m = 3
# n = 2
# print(uniquePaths(m, n))

##Time O(MN)  space O(MN)

##63. Unique Paths II
def uniquePaths2(grid):
    if grid[0][0] == 1:
        return 0
    grid[0][0] = 1
    m = len(grid)
    n = len(grid[0])

    for i in range(1, m):
        if grid[i][0] == 0:
            grid[i][0] = grid[i-1][0]
        else:
            grid[i][0] = 0

    for j in range(1, n):
        if grid[0][j] == 0:
            grid[0][j] = grid[0][j-1]
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
##Time O(MN)  space O(1)


###64. Minimum Path Sum
def minPathSum(grid):
    if len(grid) == 0 or len(grid[0]) == 0:
        return 0
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
##Time O(MN) space O(1)


##66. Plus One
def plusOne(digits):
    val = digits[-1] + 1
    carry = val // 10
    val = val%10

    digits[-1] = val
    for i in range(len(digits)-2, -1, -1):
        val = digits[i] + carry
        carry = val // 10
        val = val%10
        digits[i] = val

    if carry > 0:
        digits.insert(0, carry)
    return digits

# digits = [9, 9, 9, 9]
# print(plusOne(digits))
##Time O(N)  space O(1)

##67. Add Binary
def addBinary(a, b):
    valA = 0
    valB = 0
    a = a[::-1]
    b = b[::-1]
    for i in range(len(a)):
        if a[i] == "1":
            valA += 2**i

    for j in range(len(b)):
        if b[j] == "1":
            valB += 2**j

    val = valA + valB

    res = ""
    if val == 0:
        return "0"
    while val > 0:
        rem = val%2
        val = val // 2
        res += str(rem)

    return res[::-1]

# a = "1010"
# b = "1011"
# print(addBinary(a, b))
##Time O(M+N)  space O(1)

##69. Sqrt(x)

def sqrt(x):
    if x == 1:
        return x
    left = 0
    right = x
    while left <= right:
        mid = left + (right-left)//2
        if mid*mid == x:
            return mid
        if mid*mid < x:
            left = mid + 1
        else:
            right = mid - 1
    return left - 1

# x = 4
# print(sqrt(x))
##Time O(logN)  space O(1)

##70. Climbing Stairs'
def climbingStairs(n):
    if n == 1 or n == 0:
        return 1
    table = [0]*(n+1)
    table[1] = 1
    table[2] = 2

    for i in range(3, len(table)):
        table[i] = table[i-1] + table[i-2]
    return table[n]

# n = 5
# print(climbingStairs(n))
##Time O(n)  space O(n)

##71. Simplify Path
def simplifyPath(path):
    stack = []
    path = path.split("/")

    for i in range(1, len(path)):
        if path[i] == "" or path[i] == ".":
            continue
        if path[i] == "..":
            if len(stack) != 0:
                stack.pop()
            else:
                continue
        else:
            stack.append(path[i])

    res = "/"
    return res + "/".join(stack)

# path = "/home//foo/"
# print(simplifyPath(path))
##Time O(N)  space O(N)


##114. Flatten Binary Tree to Linked List
def flattenBTLinkedList(root):
    if root is None:
        return None
    stack = [root]
    while len(stack) != 0:
        cur = stack.pop()
        if cur.right:
            stack.append(cur.right)
        if cur.left:
            stack.append(cur.left)
        if len(stack) != 0:
            cur.right = stack[-1]
            cur.left = None

##Time O(N)  space O(N)

##116. Populating Next Right Pointers in Each Node
def nextRightPointer(root):
    if root is None:
        return None
    stack = [root]
    #root.next = None
    while len(stack) != 0:
        level = []
        for i in range(len(stack)):
            if stack[i].left:
                level.append(stack[i].left)
            if stack[i].right:
                level.append(stack[i].right)
            if i == len(stack) - 1:
                stack[i].next = None
            else:
                stack[i].next = stack[i+1]
        stack = level
    return root

##133. Clone Graph
import collections
class Node:
    def __init__(self, val, neighbours=None):
        self.val = val
        self.neighbours = neighbours if neighbours is not None else []

def cloneGraph(node):
    if node is None:
        return node
    clone = {}
    clone[node] = Node(node.val, [])

    queue = collections.deque()
    queue.append(node)

    while queue:
        cur = queue.popleft()
        for nei in cur.neighbours:
            if nei not in clone:
                clone[nei] = Node(nei.val, [])
                queue.append(nei)
            clone[cur].neighbours.append(clone[nei])
    return clone[node]

##Time O(VE)  space O(V)

###138. Copy List with Random Pointer
class listNode:
    def __init__(self, val):
        self.val = val
        self.next = None
        self.random = None

def copyListRandom(head):
    oldToCopy = {None:None}
    cur = head

    while cur:
        copy = listNode(cur.val)
        oldToCopy[cur] = copy
        cur = cur.next

    cur = head
    while cur:
        copy = oldToCopy[cur]
        copy.next = oldToCopy[cur.next]
        copy.random = oldToCopy[cur.random]
        cur = cur.next
    return oldToCopy[head]

##Time O(N)  space O(N)


###141. Linked List Cycle

def linkedListCycle(head):
    if head is None:
        return False
    slow = head
    fast = head

    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next
        if slow == fast:
            return True
    return False
##Time O(N)  space O(1)


##142. Linked List Cycle II'
def linkedListCycle2(head):
    if head is None:
        return None
    slow = head
    fast = head

    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next
        if slow == fast:
            slow = head
            while slow != fast:
                slow = slow.next
                fast = fast.next
            return slow
    return None

##Time O(N)  space O(1)


###146. LRU Cache
class DLL:
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.next = None
        self.prev = None

class LRU:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.head = DLL(0,0)
        self.tail = DLL(0,0)
        self.head.next = self.tail
        self.tail.prev = self.head


    def addNode(self, node):
        p = self.tail.prev
        p.next = node
        node.next = self.tail
        self.tail.prev = node
        node.prev = p

    def removeNode(self, node):
        p = node.prev
        n = node.next
        p.next = n
        n.prev = p

    def get(self, key):
        if key in self.cache:
            node = self.cache[key]
            self.removeNode(node)
            self.addNode(node)
            return node.val
        return -1

    def put(self, key, val):
        if key in self.cache:
            node = self.cache[key]
            self.removeNode(node)
        self.cache[key] = DLL(key, val)
        self.addNode(self.cache[key])
        if len(self.cache) > self.capacity:
            lru = self.head.next
            self.removeNode(lru)
            del self.cache[lru.key]

##Time get O(1)  put O(1)  space O(1)

# lru = LRU(2)
# lru.put(1,1)
# lru.put(2,2)
# print(lru.get(1))
# lru.put(3,3)
# print(lru.get(2))
# lru.put(4,4)
# print(lru.get(1))
# print(lru.get(3))


##149. Max Points on a Line
def maxPoints(points):
    res = 0
    while points:
        curPoint = points.pop()
        res = max(res, maxPointHelper(points, curPoint))
    return res


def maxPointHelper(points, curPoint):
    ans = 0
    slopes = {}
    x1, y1 = curPoint
    for x2, y2 in points:
        slope = (x2-x1)/(y2-y1) if y1 != y2 else 'inf'
        if slope in slopes:
            slopes[slope] += 1
        else:
            slopes[slope] = 1
        ans = max(ans, slopes[slope])
    return ans + 1

# points = [[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]]
# print(maxPoints(points))
##Time O(N^2)  space O(N)


##150. Evaluate Reverse Polish Notation
def evalRPN(tokens):
    stack = []
    for t in tokens:
        if t in "+-*/":
            a = stack.pop()
            b = stack.pop()
            if t == "+":
                res = a + b
            elif t == "-":
                res = b - a
            elif t == "*":
                res = a*b
            else:
                res = int(b/a)
            stack.append(res)
        else:
            stack.append(int(t))
    return stack[0]

# tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
# print(evalRPN(tokens))
##Time O(N)  space O(N)

##151. Reverse Words in a String
# Input: s = "the sky is blue"
# Output: "blue is sky the"

def reverseWordInStr(s):
    s = s.split()
    left = 0
    right = len(s) - 1
    while left <= right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1
    return " ".join(s)

# s = "the sky is blue"
# print(reverseWordInStr(s))
##Time O(N)  space O(1)


###152. Maximum Product Subarray
def maxProdSubArray(nums):
    prevMin = nums[0]
    prevMax = nums[0]

    res = nums[0]

    for i in range(1, len(nums)):
        curMin = min(prevMin*nums[i], prevMax*nums[i], nums[i])
        curMax = max(prevMin*nums[i], prevMax*nums[i], nums[i])

        res = max(res, curMax)

        prevMin = curMin
        prevMax = curMax

    return res

# nums = [-2,0,-1]
# print(maxProdSubArray(nums))
##Time O(N)  space O(1)

##155. Min Stack
class MinStack:
    def __init__(self):
        self.stack = []
        self.stackMin = []

    def push(self, val):
        self.stack.append(val)
        if len(self.stackMin) == 0:
            self.stackMin.append(val)
        else:
            if val > self.stackMin[-1]:
                self.stackMin.append(self.stackMin[-1])
            else:
                self.stackMin.append(val)
    def pop(self):
        self.stack.pop()
        self.stackMin.pop()

    def top(self):
        return self.stack[-1]

    def getMin(self):
        return self.stackMin[-1]


##Time All method in O(1) operation

##121. Best Time to Buy and Sell Stock
# Input: prices = [7,1,5,3,6,4]
# Output: 5


def buyAndSellStock(prices):
    res = 0
    minPrice = prices[0]
    for i in range(1, len(prices)):
        if prices[i] < minPrice:
            minPrice = prices[i]
        else:
            profit = prices[i] - minPrice
            res = max(res, profit)
    return res

# prices = [7,6,4,3,1]
# print(buyAndSellStock(prices))
##Time O(N)  space O(1)


##122. Best Time to Buy and Sell Stock II
def buyAndSellStock2(nums):
    res = 0
    for i in range(1, len(nums)):
        if nums[i-1] < nums[i]:
            profit = nums[i] - nums[i-1]
            res += profit

    return res
# nums = [7,6,4,3,1]
# print(buyAndSellStock2(nums))
##Time O(N) space O(1)

###160. Intersection of Two Linked Lists
def intersectionOfTwo(headA, headB):
    p1 = headA
    p2 = headB

    if p1 is None and p2 is None:
        return None
    while p1 or p2:
        if p1 is None and p2 != None:
            p1 = headB
        elif p1 != None and p2 is None:
            p2 = headA
        elif p1 == p2:
            return p1
        else:
            p1 = p1.next
            p2 = p2.next

##Time O(N)  space O(1)



###162. Find Peak Element
def findPick(nums):
    left = 0
    right = len(nums) - 1

    while left < right:
        mid = left + (right-left)//2
        if nums[mid] < nums[mid+1]:
            left = mid + 1
        else:
            right = mid
    return left

# nums = [1,2,1,3,5,6,4]
# print(findPick(nums))
##Time O(LogN)  space O(1)


##167. Two Sum II - Input Array Is Sorted
def twoSum2Sorted(nums, target):
    left = 0
    right = len(nums) - 1
    while left < right:
        if nums[left] + nums[right] == target:
            return [left+1, right+1]
        if nums[left] + nums[right] < target:
            left += 1
        else:
            right -= 1
    return None

# nums = [2,7,11,15]
# target = 9
# print(twoSum2Sorted(nums, target))

##169. Majority Element
##Boyers moores algorithm
def majorityElement(nums):
    element = None
    count = 0

    for num in nums:
        if count == 0:
            element = num
        if num == element:
            count += 1
        else:
            count -= 1
    return element

# nums = [2,2,1,1,1,2,2]
# print(majorityElement(nums))
##Time O(N)  space O(1)


###173. Binary Search Tree Iterator
class BSTIterator:
    def __init__(self, root):
        self.root = root
        self.stack = []
        self.partialInOrder(root)

    def partialInOrder(self, root):
        while root:
            self.stack.append(root)
            root = root.left

    def next(self):
        top = self.stack.pop()
        self.partialInOrder(top.right)
        return top.val

    def hasNext(self):
        return len(self.stack) != 0

##both next and hasNext method takes O(1) time and space O(LogN)

###207. Course Schedule
def courseSchedule(numCourses, preRequisites):
    if len(preRequisites) == 0:
        return True
    graph = {}
    for i in range(numCourses):
        graph[i] = []
    for u, v in preRequisites:
        graph[v].append(u)
    visited = [0]*numCourses
    for i in range(numCourses):
        if visited[i] == 0:
            if cycle(visited, i, graph):
                return False
    return True


def cycle(visited, i, graph):
    if visited[i] == 2:
        return True
    visited[i] = 2
    for nei in graph[i]:
        if visited[nei] != 1:
            if cycle(visited, nei, graph):
                return True
    visited[i] = 1
    return False

# numCourses = 2
# preRequisites = [[1,0],[0,1]]
# print(courseSchedule(numCourses, preRequisites))
##Time O(V+E)  space O(V+E)

##208. Implement Trie (Prefix Tree)
class TrieNode:
    def __init__(self):
        self.children = {}
        self.isEnd = False

class Trie:
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

# trie = Trie()
# trie.insert("apple")
# print(trie.search("apple"))
# print(trie.search("app"))
# print(trie.startsWith("app"))
# trie.insert('app')
# print(trie.search("app"))


###209. Minimum Size Subarray Sum
def minSizeSubArraySum(nums, target):
    n = len(nums)
    cSum = 0
    left = 0
    minLen = n + 1
    for right in range(len(nums)):
        cSum += nums[right]
        while left <= right and cSum >= target:
            minLen = min(minLen, right - left + 1)
            cSum -= nums[left]
            left += 1
    if minLen == n + 1:
        return 0
    else:
        return minLen

# target = 11
# nums = [1,1,1,1,1,1,1,1]
# print(minSizeSubArraySum(nums, target))
##Time O(N)  space O(1)


###17. Letter Combinations of a Phone Number
# Input: digits = "23"
# Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]

def letterCombination(digits):
    strs = {"2":"abc", "3":"def", "4":"ghi", "5":"jkl", "6":"mno", "7":"pqrs", "8":"tuv", "9":"wxyz"}
    res = []
    if len(digits) == 0:
        return res
    newStr = []
    backtrack(strs, digits, 0, newStr, res)
    return res

def backtrack(strs, digits, i, newStr, res):
    if i == len(digits):
        res.append("".join(newStr))
        return
    cur = strs[digits[i]]
    for k in range(len(cur)):
        newStr.append(cur[k])
        backtrack(strs, digits, i+1, newStr, res)
        newStr.pop()

# digits = "23"
# print(letterCombination(digits))
##Time O(n*4^n)  space O(4^n)


##22. Generate Parentheses
def generateParentheses(n):
    res = []
    stack = []

    def backtracking2(openParen, closeParen):
        if openParen == closeParen == n:
            res.append("".join(stack))
            return
        if openParen < n:
            stack.append("(")
            backtracking2(openParen+1, closeParen)
            stack.pop()

        if openParen > closeParen:
            stack.append(")")
            backtracking2(openParen, closeParen+1)
            stack.pop()

    backtracking2(0,0)
    return res
# n = 3
# print(generateParentheses(n))
##Time O(4^n/sqrt(n))  space O(4^n/sqrt(n))

###46. Permutations
# Input: nums = [1,2,3]
# Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

def permutations(nums):
    res = []
    temp = []
    if len(nums) == 0:
        return res
    if len(nums) == 1:
        return [nums]
    def DFS(temp):
        if len(temp) == len(nums):
            res.append(temp)
            return
        for num in nums:
            if num not in temp:
                DFS(temp+[num])
    DFS(temp)
    return res

# nums = [1,2,3]
# print(permutations(nums))
##Time O(n!)  space O(N)


###=======================Can SUM problem===============
# nums = [2,4,6,3]
# target = 10

# 0 1 2 3 4 5 6 7 8 9 10
# T F T T T T T T T T T


def canSum(nums, target):
    table = [False]*(target+1)
    table[0] = True

    for i in range(len(table)):
        if table[i] == True:
            for num in nums:
                if i + num <= len(table) - 1:
                    table[i+num] = True
    return table[-1]

# nums = [0, 3]
# target = 19
# print(canSum(nums, target))
##Time O(MN) space O(M)  M = target sum, N = length of nums

###====================How Sum=====================
# 2,5,3
#7
# 0  1  2    3    4         5          6        7
# []   [2]  [3]  [2,2]     [3,2]     [4,2]      [5,2]

def howSum(nums, target):
    if target == 0:
        return []
    table = [None]*(target+1)
    table[0] = []
    for i in range(len(table)):
        if table[i] != None:
            for num in nums:
                if i + num <= len(table) - 1:
                    table[i+num] = table[i] + [num]
    return table[-1] if table[-1] != None else "Not possible"

# nums = [7,3]
# target = 839
# print(howSum(nums, target))
##Time O(M^2*N)  space O(M^2)  M = targetn N = length of nums

##==================Best Sum===================================================

def bestSum(nums, target):
    if target == 0:
        return []
    table = [None]*(target+1)
    table[0] = []
    for i in range(len(table)):
        if table[i] != None:
            for num in nums:
                if i + num <= len(table) - 1:
                    combination = table[i] + [num]
                    if table[i+num] == None or len(combination) < len(table[i+num]):
                        table[i+num] = combination
    return table[-1]

# nums = [2,3,4,5]
# target = 20
# print(bestSum(nums, target))
##Time O(M^2*N)  space O(M^2)  M = Target, N = Lenght of nums

###39. Combination Sum
# Input: candidates = [2,3,6,7], target = 7
# Output: [[2,2,3],[7]]

def combinationSum(nums, target):
    res = []
    temp = []
    helper_comSum(nums, target, res, temp)
    return res

def helper_comSum(nums, target, res, temp):
    if target == 0:
        res.append(temp)
        return
    if target < 0:
        return

    for i, num in enumerate(nums):
        helper_comSum(nums[i:], target-num, res, temp + [num])


# nums = [2,3,5]
# target = 8
# print(combinationSum(nums, target))
##Time O(2^N)  space O(N)


##40. Combination Sum II
def comSum2(nums, target):
    res = []
    temp = []
    nums.sort()
    helper_comSum2(0, nums, target, res, temp)
    return res

def helper_comSum2(start, nums, target, res, temp):
    if target == 0:
        res.append(temp)
        return
    if target < 0:
        return
    for i in range(start, len(nums)):
        if i > start and nums[i] == nums[i-1]:
            continue
        helper_comSum2(i, nums, target-nums[i], res, temp+[nums[i]])

# nums = [2,3,5]
# target = 8
# print(comSum2(nums, target))
##Time O(2^n)  space O(N)

##219. Contains Duplicate II
def containDuplicate2(nums, k):
    dict1 = {}
    for j in range(len(nums)):
        if nums[j] in dict1:
            i = dict1[nums[j]]
            if abs(i-j) <= k:
                return True
        dict1[nums[j]] = j
    return False

# nums = [1,2,3,1,2,3]
# k = 2
# print(containDuplicate2(nums, k))

##Time O(N)  space O(N)

###221. Maximal Square
def maximalSquare(grid):
    if len(grid) == 0 or len(grid[0]) == 0:
        return 0
    m = len(grid)
    n = len(grid[0])

    table = [[0 for col in range(n+1)] for row in range(m+1)]
    area = 0
    for i in range(1, len(table)):
        for j in range(1, len(table[0])):
            if grid[i-1][j-1] == "1":
                table[i][j] = 1 + min(table[i-1][j], table[i][j-1], table[i-1][j-1])
                height = table[i][j]
                area = max(area, height*height)

    return area

# grid = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
# print(maximalSquare(grid))
##Time O(MN)  space O(MN)


##222. Count Complete Tree Nodes

def countCompleteTreeNode(root):
    if root is None:
        return 0
    leftHeight = 1
    rightHeight = 1

    while root.left:
        leftHeight += 1
        root = root.left

    while root.right:
        rightHeight += 1
        root = root.right

    if leftHeight == rightHeight:
        return 2**leftHeight - 1
    return 1 + countCompleteTreeNode(root.left) + countCompleteTreeNode(root.right)

##Time O(Logn)  space O(logn)

##226. Invert Binary Tree
def invertBT(root):
    if root is None:
        return None
    root.left, root.right = root.right, root.left
    invertBT(root.left)
    invertBT(root.right)
    return root

##Time O(N)  space O(N)

###230. Kth Smallest Element in a BST
def kThSmallestBST(root, k):
    if root is None:
        return None
    count = 1
    stack = []

    while stack or root:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        count += 1
        if count == k:
            return root.val
        root = root.right

##Time O(logn) average   O(N) worst  space O(N)


##234. Palindrome Linked List
def palindromLinkedList(head):
    if head is None or head.next is None:
        return True
    slow = head
    fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    if fast:
        slow = slow.next

    mid = slow
    cur = head

    last = reverse(mid)
    while last:
        if cur.val != last.val:
            return  False
        last = last.next
        cur = cur.next
    return True

def reverse(root):
    prev = None
    cur = root
    while cur:
        temp = cur.next
        cur.next = prev
        prev = cur
        cur = temp
    return prev

##Time O(N)  space O(1)


###235. Lowest Common Ancestor of a Binary Search Tree
def lowestCommonAncestorBST(root, p, q):
    ##Iterative approach
    while root:
        if root.val > p and root.val > q:
            root = root.left
        elif root.val < p and root.val < q:
            root = root.right
        else:
            return root

##Time O(N)  space O(1)

    # if root == p or root == q:
    #     return root
    # if root.val > p and root.val > q:
    #     return lowestCommonAncestorBST(root.left, p, q)
    # if root.val < p and root.val < q:
    #     return lowestCommonAncestorBST(root.right, p, q)


###237. Delete Node in a Linked List
## 1 2 4 4
def deleteNodeInLinkedList(head, node):
    node.val = node.next.val
    node.next = node.next.next
    return head



###238. Product of Array Except Self

# nums = [1,2,3,4]
#
# 1 2 3 4
#
# [24,12,8,6]
#
# left = 1 2 6 24
# product = 1
#       12  8  6


def productOfArrayExceptSelf(nums):
    res = []
    prod = 1
    for i in range(len(nums)):
        prod *= nums[i]
        res.append(prod)
    print(res)
    prod = 1
    for i in range(len(nums)-1, 0, -1):
        res[i] = prod*res[i-1]
        prod *= nums[i]
    res[0] = prod
    return res

# nums = [-1,1,0,-3,3]
# print(productOfArrayExceptSelf(nums))

##Time O(N)  space O(1)


##240. Search a 2D Matrix II
def searchIn2DGrid2(grid, target):
    if len(grid) == 0 or len(grid[0]) == 0:
        return False
    row = 0
    col = len(grid[0]) - 1

    while row < len(grid) and col >= 0:
        if grid[row][col] == target:
            return True
        if grid[row][col] > target:
            col -= 1
        else:
            row += 1

    return False

##Time O(M+N)  space O(1)



###257. Binary Tree Paths
def binaryTreePath(root):
    if root is None:
        return []
    res = []
    stack =[(root, [str(root.val)])]
    while len(stack) != 0:
        node, path = stack.pop()
        if node.left is None and node.right is None:
            res.append("->".join(path))
        if node.left:
            stack.append((node.left, path+[str(node.left)]))
        if node.right:
            stack.append((node.right, path+[str(node.right)]))
    return res

##Time O(N)  space O(N)

##263. Ugly Number
def uglyNumber(num):
    if num <= 0:
        return False
    if num == 1:
        return True
    while num > 1:
        if num % 2 == 0:
            num = num // 2
        elif num % 3 == 0:
            num = num // 3
        elif num % 5 == 0:
            num = num / 5
        else:
            return False
    return True

###Ugly Number II
def uglyNumber2(num):
    table = [0]*num
    table[0] = 1
    i2 = i3 = i5 = 0
    mul2 = 2
    mul3 = 3
    mul5 = 5

    for i in range(1, len(table)):
        table[i] = min(mul2, mul3, mul5)
        if table[i] == mul2:
            i2 += 1
            mul2 = table[i2]*2
        if table[i] == mul3:
            i3 += 1
            mul3 = table[i3]*3
        if table[i] == mul5:
            i5 += 1
            mul5 = table[i5]*5
    return table[-1]

# num = 10
# print(uglyNumber2(num))
##Time O(1)  space O(1)
# 1 2 3 4 5 6 8 9 10 12
#
# i2 = 6
# i3 = 3
# i5 = 2
#
# mul2 = 16
# mul3 = 12
# mul5 = 15


##287. Find the Duplicate Number
def findDuplicate(nums):
    slow = nums[0]
    fast = nums[0]

    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break
    slow = nums[0]
    while fast != slow:
        slow = nums[slow]
        fast = nums[fast]
    return slow

# nums = [3,1,3,4,2]
# print(findDuplicate(nums))
##Time O(N)  space O(1)


###289. Game of Life
def gameOfLife(grid):
    ##live --> dead 2(later make 0)
    ##dead --> live -1 (later make 1)
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 0:
                lives = countLives(grid, i, j)
                if lives == 3:
                    grid[i][j] = -1
            if grid[i][j] == 1:
                lives = countLives(grid, i, j)
                if lives < 2 or lives > 3:
                    grid[i][j] = 2
                if lives == 2 or lives == 3:
                    grid[i][j] = 1

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 2:
                grid[i][j] = 0
            if grid[i][j] == -1:
                grid[i][j] = 1
    return grid

def countLives(grid, i, j):
    dirs = [(1,0), (-1, 0), (0, 1), (0, -1), (1, 1),(-1, -1),(1, -1), (-1, 1)]
    res = 0
    for dir in dirs:
        newRow = i + dir[0]
        newCol = j + dir[1]

        if (newRow >= 0 and newRow < len(grid) and newCol >= 0 and newCol < len(grid[0])) and (grid[newRow][newCol] == 1 or grid[newRow][newCol] == 2):
            res += 1
    return res

# grid = [[1,1],[1,0]]
# print(gameOfLife(grid))
##Time O(MN) space O(1)

def countSmaller(nums):
    res = []
    num1 = nums
    for i in range(len(nums)):
        idx = partition(nums[i:], 0, len(nums[i:]) - 1)
        print(idx)
        res.append(len(nums[:idx]))
        nums = num1
    return res

def partition(nums, start, end):
    pivot = start
    nums[pivot], nums[end] = nums[end], nums[pivot]
    for i in range(start, end):
        if nums[i] < nums[end]:
            nums[i], nums[start] = nums[start], nums[i]
            start += 1
    nums[start], nums[end] = nums[end], nums[start]
    return start

# nums = [5,2,6,1]
# print(countSmaller(nums))

##55.Jump Game

def jumpGame(nums):
    maxReach = 0
    for i in range(len(nums)):
        if i > maxReach:
            return False
        maxReach = max(maxReach, i+nums[i])
    return True

# nums = [3,2,1,0,4]
# print(jumpGame(nums))

##51. N-Queens
def NQueen(n):
    if n == 1:
        return [["Q"]]
    board = [["." for col in range(n)] for row in range(n)]
    res = []
    QueenBackTrack(board, res, 0, n)
    return res

def QueenBackTrack(board, res, row, n):
    if n == row:
        res.append(construct(board))
        return
    for col in range(n):
        if canConstruct(board, row, col):
            board[row][col] = "Q"
            QueenBackTrack(board, res, row+1, n)
            board[row][col] = "."  ##backtrack

def canConstruct(board, row, col):
    ##left diag
    x = row
    y = col
    while x >= 0 and y >= 0:
        if board[x][y] == "Q":
            return False
        x -= 1
        y -= 1

    ##Right Diag
    x = row
    y = col
    while x >= 0 and y < len(board):
        if board[x][y] == "Q":
            return False
        x -= 1
        y += 1

    ##col check
    for i in range(len(board)):
        if board[i][col] == "Q":
            return False
    return True

def construct(board):
    ans = []
    for row in board:
        ans.append("".join(row))
    return ans

# n = 4
# print(NQueen(n))
##Time O(N!)  space O(N^2)

##4. Median of Two Sorted Arrays
def medianTwoSortedArray(nums1, nums2):
    if len(nums1) > len(nums2):
        return medianTwoSortedArray(nums2, nums1)
    x = len(nums1)
    y = len(nums2)

    left = 0
    right = x

    while left <= right:
        partitionX = (left+right)//2
        partitionY = (x+y+1)//2 - partitionX

        if partitionX == 0:
            maxLeftX = float('-inf')
        else:
            maxLeftX = nums1[partitionX-1]

        if partitionX == x:
            minRightX = float('inf')
        else:
            minRightX = nums1[partitionX]

        if partitionY == 0:
            maxLeftY = float('-inf')
        else:
            maxLeftY = nums2[partitionY-1]
        if partitionY == y:
            minRightY = float('inf')
        else:
            minRightY = nums2[partitionY]
            
        if maxLeftX <= minRightY and maxLeftY <= minRightX:
            if (x+y)%2 != 0:
                return max(maxLeftX, maxLeftY)
            else:
                return (max(maxLeftX, maxLeftY) + min(minRightX, minRightY)) / 2
        if maxLeftX < minRightY:
            left = partitionX + 1
        else:
            right = partitionX - 1

# nums1 = [1,2]
# nums2 = [3,4]
# print(medianTwoSortedArray(nums1, nums2))
##Time O(log(min(M,N))  space O(1)

##224. Basic Calculator
def basicCalculator(s):
    res = 0
    sign = 1
    number = 0
    stack = []
    for c in s:
        if c.isdigit():
            number = number*10 + int(c)
        else:
            if c == "+":
                res += sign*number
                sign = 1
                number = 0
            elif c == "-":
                res += sign*number
                sign = -1
                number = 0
            elif c == "(":
                res += sign*number
                stack.append(res)
                stack.append(sign)

                sign = 1
                number = 0
                res = 0
            elif c == ")":
                res += sign*number
                res *= stack.pop()
                res += stack.pop()

                sign = 1
                number = 0
    res = res + sign*number
    return res

# s = "(1+(4+5+2)-3)+(6+8)"
# print(basicCalculator(s))
##Time O(N)  space O(N)


##227. Basic Calculator II

def basicCalculator2(s):
    res = 0
    operation = "+"
    number = 0
    stack = []

    for i,c in enumerate(s):
        if c.isdigit():
            number = number*10 + int(c)
        if not c.isdigit() and c in "+-*/" or i == len(s) - 1:
            if operation == "+":
                stack.append(number)
                operation = c
                number = 0
            elif operation == "-":
                stack.append(-number)
                operation = c
                number = 0
            elif operation == "*":
                stack.append(stack.pop()*number)
                operation = c
                number = 0
            elif operation == "/":
                stack.append(int(stack.pop()/number))
                operation = c
                number = 0
    while len(stack) != 0:
        res += stack.pop()
    return res

# s = " 3+5 / 2 "
# print(basicCalculator2(s))
##Time O(N)  space O(N)

##367. Valid Perfect Square

def validPerfectsquare(num):
    left = 0
    right = num

    while left <= right:
        mid = (left+right)//2
        if mid*mid == num:
            return True
        if mid*mid > num:
            right = mid - 1
        else:
            left = mid + 1
    return False

# num = 81
# print(validPerfectsquare(num))
##Time O(logn)  space O(1)

###381. Insert Delete GetRandom O(1) - Duplicates allowed
import collections
from random import choice
class randCollectionDuplicate:
    def __init__(self):
        self.array = []
        self.dict1 = collections.defaultdict(set)

    def insert(self, val):
        self.dict1[val].add(len(self.array))
        self.array.append(val)
        return len(self.dict1[val]) == 1

    def remove(self, val):
        print(self.array)
        print(self.dict1)
        if val not in self.dict1:
            return False
        idx = self.dict1[val].pop()
        lastElement = self.array[-1]
        self.array[idx] = lastElement

        self.dict1[lastElement].add(idx)
        self.dict1[lastElement].remove(len(self.array)-1)
        self.array.pop()
        return True

    def getRandom(self):
        return choice(self.array)

# ran = randCollectionDuplicate()
# print(ran.insert(1))
# print(ran.insert(1))
# print(ran.insert(2))
# print(ran.getRandom())
# print(ran.remove(1))
# print(ran.getRandom())

##387. First Unique Character in a String
def firstUniChar(s):
    dictS = {}
    for c in s:
        if c in dictS:
            dictS[c] += 1
        else:
            dictS[c] = 1

    for i in range(len(s)):
        if dictS[s[i]] == 1:
            return i
    return -1

# s = "aabb"
# print(firstUniChar(s))

###329. Longest Increasing Path in a Matrix

def longestIncreasingPath(grid):
    if len(grid) == 0:
        return 0
    maxLen = 0
    memo = {}
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            maxLen = max(maxLen, longestPathHelper(grid, i, j, memo))
    return maxLen


def longestPathHelper(grid, i, j, memo):
    length = 1
    dirs = [(1,0), (0,1), (-1, 0), (0, -1)]
    if (i,j) in memo:
        return memo[(i,j)]
    for d in dirs:
        newRow = i + d[0]
        newCol = j + d[1]

        if newRow >= 0 and newRow < len(grid) and newCol >= 0 and newCol < len(grid[0]) and grid[newRow][newCol] > grid[i][j]:
            length = max(length, 1 + longestPathHelper(grid, newRow, newCol, memo))
    memo[(i,j)] = length
    return length

# grid = [[3,4,5],[3,2,6],[2,2,1]]
# print(longestIncreasingPath(grid))
##Time O(MN)  space O(MN)


##328. Odd Even Linked List
def oddEvenLinkedList(head):
    if head is None:
        return None
    odd = head
    even = head.next

    oddHead = head
    evenHead = head.next

    while evenHead and evenHead.next:
        evenHead.next = evenHead.next.next
        oddHead.next = oddHead.next.next

        evenHead = evenHead.next
        oddHead = oddHead.next

    oddHead.next = even
    return head

##Time O(N)  space O(1)


##322. Coin Change

def coinChange(coins, amount):
    if amount == 0:
        return 0
    table = [amount+1]*(amount+1)
    table[0] = 0
    for val in range(len(table)):
        for coin in coins:
            if val - coin >= 0:
                table[val] = min(table[val], 1 + table[val-coin])
    print(table)
    if table[-1] > amount:
        return -1
    else:
        return table[-1]

# coins = [2]
# amount = 3
# print(coinChange(coins, amount))
# ##Time O(MN)  space O(M)  M = amount  N = number of coins

##310. Minimum Height Trees
def minHeightTree(n, edges):
    if n <= 2:
        return [i for i in range(n)]
    graph = {}
    for i in range(n):
        graph[i] = []
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    leaves = []
    for node in range(n):
        if len(graph[node]) == 1:
            leaves.append(node)

    remaining = n
    while remaining > 2:
        remaining -= len(leaves)
        newLeaves = []
        while leaves:
            leaf = leaves.pop()
            nei = graph[leaf].pop()
            graph[nei].remove(leaf)
            if len(graph[nei]) == 1:
                newLeaves.append(nei)
        leaves = newLeaves
    return leaves


# n = 6
# edges = [[3,0],[3,1],[3,2],[3,4],[5,4]]
# print(minHeightTree(n, edges))
##Time O(V)  space O(V)

##91. Decode Ways
def decodeWays(s):
    if len(s) == 0:
        return 0
    n = len(s)
    table = [0]*(n+1)

    table [0] = 1
    for i in range(1, n+1):
        if s[i-1] != '0':
            table[i] += table[i-1]
        if i > 1 and '10' <= s[i-2:i] <= '26':
            table[i] += table[i-2]
    return table[-1]

# s = "067"
# print(decodeWays(s))
##Time O(N)  space O(N)

##264. Ugly Number II
def uglyNum2(n):
    ugly = [0]*n
    ugly[0] = 1

    i2 = 0
    i3 = 0
    i5 = 0

    mul2 = 2
    mul3 = 3
    mul5 = 5

    for i in range(1, n):
        ugly[i] = min(mul2, mul3, mul5)

        if ugly[i] == mul2:
            i2 += 1
            mul2 = ugly[i2]*2

        if ugly[i] == mul3:
            i3 += 1
            mul3 = ugly[i3]*3

        if ugly[i] == mul5:
            i5 += 1
            mul5 = ugly[i5]*5
    print(ugly)
    return ugly[-1]

# n = 10
# print(uglyNum2(n))
##Time O(N)  space O(N)

##207. Course Schedule

def courseSche(numCourses, preRequisites):
    if len(preRequisites) == 0:
        return True
    visited = [0]*numCourses
    graph = {}
    for i in range(numCourses):
        graph[i] = []
    for u, v in preRequisites:
        graph[v].append(u)

    for i in range(numCourses):
        if visited[i] == 0:
            if isCycle(graph, visited, i):
                return False
    return True

def isCycle(graph, visited, i):
    if visited[i] == 2:
        return True
    visited[i] = 2
    for nei in graph[i]:
        if visited[nei] != 1:
            if isCycle(graph, visited, nei):
                return True
    visited[i] = 1
    return False

# numCourses = 2
# prerequisites = [[1,0],[0,1]]
# print(courseSche(numCourses, prerequisites))
##Time O(V+E)  space O(V+E)

###210. Course Schedule II

def courseSche2(numCourses, preRequisites):
    res = []
    graph = {}
    for i in range(numCourses):
        graph[i] = []
    for u, v in preRequisites:
        graph[v].append(u)

    if detectCycle(numCourses, graph):
        return res
    visited = [False]*numCourses

    stack = []
    for node in range(numCourses):
        if not visited[node]:
            courseScDFS(graph, visited, node, stack)

    return stack[::-1]

def courseScDFS(graph, visited, node, stack):
    visited[node] = True
    for nei in graph[node]:
        if not visited[nei]:
            courseScDFS(graph, visited, nei, stack)
    stack.append(node)

def detectCycle(numCourses, graph):
    visited = [0]*numCourses
    for node in range(numCourses):
        if visited[node] == 0:
            if isCyc(graph, visited, node):
                return True
    return False

def isCyc(graph, visited, node):
    if visited[node] == 2:
        return True
    visited[node] = 2
    for nei in graph[node]:
        if visited[nei] != 1:
            if isCyc(graph, visited, nei):
                return True
    visited[node] = 1
    return False

# numCourses = 4
# prerequisites = [[1,0],[2,0],[3,1],[3,2]]
# print(courseSche2(numCourses, prerequisites))
##Time O(V+E)  space O(V+E)

###203. Remove Linked List Elements
class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None

def removeElements(head, val):
    if root is None:
        return None
    prev = ListNode(0)
    prev.next = head
    cur = head
    record = prev

    while cur:
        if cur.val == val:
            prev.next = cur.next
            cur = cur.next
        else:
            prev = cur
            cur = cur.next

    return record.next


##524. Longest Word in Dictionary through Deleting
# Input: s = "abpcplea", dictionary = ["ale","apple","monkey","plea"]
# Output: "apple"


def longestWordInDict(s, dict1):
    res = ""
    for word in dict1:
        if isSubsequence(s, word):
            if len(word) > len(res) or (len(res)==len(word) and res > word):
                res = word
    return res

def isSubsequence(s, word):
    if len(word) > len(s):
        return False
    i = 0
    j = 0

    while i < len(s) and j < len(word):
        if s[i] == word[j]:
            i += 1
            j += 1
        else:
            i += 1
    return j == len(word)

# s = "abpcplea"
# dict1 = ["ale","apple","monkey","plea"]
# print(longestWordInDict(s, dict1))
##Time O(S*N) space O(1)  S = s, N = length of dict1

###523. Continuous Subarray Sum
def continuousSubArraySum(nums, k):
    cSum = 0
    dict1 = {0:-1}
    for i, num in enumerate(nums):
        cSum += num
        cSum = cSum%k

        if cSum in dict1:
            if i-dict1[cSum] >= 2:
                return True
        else:
            dict1[cSum] = i
    return False

# nums = [23,2,6,4,7]
# k = 13
# print(continuousSubArraySum(nums, k))
##Time O(N)  space O(N)

###516. Longest Palindromic Subsequence
def longPalinSubsequence(s):
    if len(s) < 2:
        return len(s)
    table = [[0 for col in range(len(s))] for row in range(len(s))]
    for i in range(len(table)):
        table[i][i] = 1

    for i in range(len(s)-2, -1, -1):
        for j in range(i+1, len(s)):
            if s[i] == s[j]:
                table[i][j] = 2 + table[i+1][j-1]
            else:
                table[i][j] = max(table[i+1][j], table[i][j-1])
    return table[0][-1]

# s = "cbbd"
# print(longPalinSubsequence(s))
##Time O(MN)  space O(MN)


def longPalinSubsequenceTest(s):
    if len(s) < 2:
        return s
    table = [["" for col in range(len(s))] for row in range(len(s))]
    for i in range(len(table)):
        table[i][i] = s[i]

    for i in range(len(s)-2, -1, -1):
        for j in range(i+1, len(s)):
            if s[i] == s[j]:
                table[i][j] = s[i] + table[i+1][j-1] + s[j]
            else:
                if len(table[i+1][j]) >= len(table[i][j-1]):
                    table[i][j] = table[i+1][j]
                else:
                    table[i][j] = table[i][j-1]
    return table[0][-1]

# s = "cbbd"
# print(longPalinSubsequenceTest(s))

###513. Find Bottom Left Tree Value
def leftTreeValue(root):
    if root is None:
        return None
    stack = [root]
    res = root.val

    while stack:
        temp = []
        for node in stack:
            if node.left:
                temp.append(node.left)
            if node.right:
                temp.append(node.right)
        if len(temp) > 0:
            res = temp[0].val
        stack = temp

    return res


###501. Find Mode in Binary Search Tree
def findMode(root):
    if root is None:
        return None
    freq = 0
    maxFreq = 0
    res = []
    prev = None

    def inOrder(root):
        if root:
            if root.left:
                inOrder(root.left)

            if prev.val == root.val:
                freq += 1
            else:
                freq = 1
            prev = root
            if freq > maxFreq:
                maxFreq = freq
                res = [root.val]
            elif freq == maxFreq:
                res.append(root.val)
            if root.right:
                inOrder(root.right)
    inOrder(root)
    return res

##Time O(N) space O(logn)


###496. Next Greater Element I

# Input: nums1 = [4,1,2], nums2 = [1,3,4,2]
# Output: [-1,3,-1]

def nextGreater1(nums1, nums2):
    if len(nums1) == 0 or len(nums2) == 0:
        return []
    res = []
    dict1 = {}
    for num in nums2:
        dict1[num] = -1
    stack = []
    for i in range(len(nums2)):
        while len(stack) != 0 and stack[-1] < nums2[i]:
            dict1[stack[-1]] = nums2[i]
            stack.pop()
        stack.append(nums2[i])

    for num in nums1:
        res.append(dict1[num])
    return res

# nums1 = [2,4]
# nums2 = [1,2,3,4]
# print(nextGreater1(nums1, nums2))
##Time O(M+N) space O(M)  M = length of nums2 N = length of nums1

####503. Next Greater Element II
# Input: nums = [1,2,3,4,3]
# Output: [2,3,4,-1,4]

def nextGreater2(nums):
    n = len(nums)
    res = [-1]*n
    stack = []
    for i in range(len(nums)):
        while len(stack) != 0 and nums[i] > nums[stack[-1]]:
            res[stack[-1]] = nums[i]
            stack.pop()
        stack.append(i)

    for i in range(len(nums)):
        ##Early stopping
        if i == stack[-1]:
            break
        while len(stack) != 0 and nums[i] > nums[stack[-1]]:
            res[stack[-1]] = nums[i]
            stack.pop()
    return res

# nums = [1,2,3,4,3]
# print(nextGreater2(nums))
##Time O(N)  space O(N)


##695. Max Area of Island
def maxAreaOfIsland(grid):
    maxArea = 0
    visited = [[False for col in range(len(grid[0]))] for row in range(len(grid))]

    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1 and not visited[i][j]:
                area = islandDFS(grid, i, j, visited)
                maxArea = max(maxArea, area)
    return maxArea

def islandDFS(grid, i, j, visited):
    area = 1
    if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] != 1:
        return 0
    if visited[i][j]:
        return 0
    visited[i][j] = True
    area += islandDFS(grid, i+1, j, visited)
    area += islandDFS(grid, i-1, j, visited)
    area += islandDFS(grid, i, j+1, visited)
    area += islandDFS(grid, i, j-1, visited)
    return area

#grid = [[0,0,1,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,1,1,0,1,0,0,0,0,0,0,0,0],[0,1,0,0,1,1,0,0,1,0,1,0,0],[0,1,0,0,1,1,0,0,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,1,1,0,0,0,0]]
# grid = [[0,0,0,0,0,0,0,0]]
# print(maxAreaOfIsland(grid))
##Time O(MN)  space O(MN)


###692. Top K Frequent Words
# Input: words = ["i","love","leetcode","i","love","coding"], k = 2
# Output: ["i","love"]

def topKfrequent(words, k):
    wordDict = {}
    for word in words:
        if word in wordDict:
            wordDict[word] += 1
        else:
            wordDict[word] = 1

    maxHeap = []
    print(wordDict)
    import heapq
    for word, val in wordDict.items():
        heapq.heappush(maxHeap, (-val, word))
    print(maxHeap)
    res = []
    while k > 0:
        ans = heapq.heappop(maxHeap)
        print(ans)
        res.append(ans[1])
        k -= 1
    return res
# words = ["i","love","leetcode","i","love","coding"]
# k = 2
# print(topKfrequent(words, k))

###680. Valid Palindrome II
def validPalin2(s):
    if len(s) == 1:
        return True
    left = 0
    right = len(s) - 1

    while left <= right:
        if s[left] != s[right]:
            if not isPalin(s, left+1, right) or not isPalin(s, left, right-1):
                return False
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

# s = "abc"
# print(validPalin2(s))
##Time O(N)  space O(1)4


###676. Implement Magic Dictionary
class magicDictionary:
    def __init__(self):
        self.dict1 = {}

    def buildDict(self, dictionary):
        for word in dictionary:
            if len(word) in self.dict1:
                self.dict1[len(word)].append(word)
            else:
                self.dict1[len(word)] = [word]

    def search(self, searchWord):
        if len(searchWord) not in self.dict1:
            return False
        wordList = self.dict1[len(searchWord)]
        for word in wordList:
            count = 0
            for i in range(len(word)):
                if word[i] != searchWord[i]:
                    count += 1
                if count > 1:
                    break
            if count == 1:
                return True
        return False

##Time bulidDict() --> O(N) N -- length of dictionary  search() --> O(MN)

###674. Longest Continuous Increasing Subsequence

def longestConIncreSubSe(nums):
    if len(nums) == 1:
        return 1
    maxLen = 1
    left = 0
    length = 1
    for right in range(1, len(nums)):
        if nums[right] > nums[right-1]:
            length += 1
            maxLen = max(maxLen, length)
        else:
            length = 1
    return maxLen

# nums = [2,2,2,2,2]
# print(longestConIncreSubSe(nums))
##Time O(N)  space O(1)


###673. Number of Longest Increasing Subsequence
# Input: nums = [1,3,5,4,7]
# Output: 2
# Explanation: The two longest increasing subsequences are [1, 3, 4, 7] and [1, 3, 5, 7].

def numOfIncreasingSub(nums):
    n = len(nums)
    LIS = [1]*n
    count = [1]*n

    for i in range(1, n):
        for j in range(i):
            if nums[i] > nums[j]:
                if LIS[j] >= LIS[i]:
                    LIS[i] = LIS[j] + 1
                    count[i] = count[j]
                elif LIS[i] == LIS[j] + 1:
                    count[i] += count[j]
    ans = 0
    maxLen = max(LIS)
    for i in range(n):
        if LIS[i] == maxLen:
            ans += count[i]
    return ans

# nums = [2,2,2,2,2]
# print(numOfIncreasingSub(nums))
##Time O(N^2)  space O(N)

###671. Second Minimum Node In a Binary Tree
def secondMinBT(root):
    ans = float('inf')
    Min = root.val

    def DFS(root):
        if root:
            if Min < root.val < ans:
                ans = root.val
            DFS(root.left)
            DFS(root.right)
    DFS(root)
    if ans == float('inf'):
        return -1
    else:
        return ans
##Time O(N)

####670. Maximum Swap
def maximumSwap(num):
    numList = [int(c) for c in str(num)]
    #   print(numList)
    dict1 = {}
    for i,num1 in enumerate(numList):
        dict1[num1] = i
    print(dict1)
    for i, num1 in enumerate(numList):
        for n in range(9,num1,-1):
            if n in dict1 and dict1[n] > i:
                numList[i], numList[dict1[n]] = numList[dict1[n]], numList[i]
                print(numList)
                return int("".join([str(d) for d in numList]))
    return num


# num = 9973
# print(maximumSwap(num))
##Time O(N)  space O(N)


####668. Kth Smallest Number in Multiplication Table
def kthSmallestMulTable(m, n, k):
    left = 1
    right = m*n

    while left <= right:
        mid = left + (right-left)//2
        count = mulTableCount(mid, m, n)
        if count >= k:
            right = mid - 1
        else:
            left = mid + 1
    return left

def mulTableCount(mid, m, n):
    temp = 0
    for i in range(1, m+1):
        temp += min(mid//i, n)
    return temp

# m = 2
# n = 3
# k = 6
# print(kthSmallestMulTable(m, n, k))
##Time O(mlogmn)  space O(1)

##665. Non-decreasing Array
def nonDecreasingArray(nums):
    pos = -1
    for i in range(len(nums)-1):
        if nums[i] > nums[i+1]:
            if pos != -1:
                return False
            pos = i
    return pos == -1 or pos == 0 or pos == len(nums) -1 or nums[pos-1] <= nums[pos+1] or nums[pos] <= nums[pos+2]

# nums = [4,2,1]
# print(nonDecreasingArray(nums))
##Time O(N)  space O(1)


##658. Find K Closest Elements

# Input: arr = [1,2,3,4,5], k = 4, x = 3
# Output: [1,2,3,4]

def kClosestElements(arr, k, x):
    left = 0
    right = len(arr) - k

    while left < right:
        mid = left + (right-left)//2
        if x - arr[mid] > arr[mid+k] - x:
            left = mid + 1
        else:
            right = mid
    return arr[left:left+k]

# arr = [1,2,3,4,5]
# k = 4
# x = -1
# print(kClosestElements(arr, k, x))
##Time O(Log(n-k))  space O(1)

###653. Two Sum IV - Input is a BST
def twoSum4InputBST(root, target):
    if root is None:
        return False
    stack = []
    res = []
    while root or stack:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        res.append(root.val)
        root = root.right

    left = 0
    right = len(res)

    while left < right:
        if res[left] + res[right] == target:
            return True
        if res[left] + res[right] < target:
            left += 1
        else:
            right -= 1
    return False

##Time O(N)  space O(N)


###Find Duplicate Subtrees
# Input: dictionary = ["cat","bat","rat"], sentence = "the cattle was rattled by the battery"
# Output: "the cat was rat by the bat"

class TrieNode:
    def __init__(self):
        self.children = {}
        self.isEnd = True
class Trie:
    def __init__(self):
        self.root = TrieNode()
    def addWord(self, word):
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
                return word
            if parent.children[char].isEnd:
                return word[:i+1]
            parent = parent.children[char]
        return word


def replaceWords(dictionary, sentence):
    t = Trie()
    for word in dictionary:
        t.addWord(word)
    sent = sentence.split()
    for i, word in enumerate(sent):
        w = t.search(word)
        sent[i] = w
    return "".join(sent)

##Time O(MN)  space O(N)  N = total word in sentence  M = average length of words

##647. Palindromic Substrings
def palinSubStr(s):
    if len(s) == 0:
        return 0
    if len(s) == 1:
        return 1
    count = 0
    for i in range(len(s)):
        count += countPalinSubStr(s, i, i)
        count += countPalinSubStr(s, i, i+1)
    return count


def countPalinSubStr(s, left, right):
    count = 0
    while left >= 0 and right < len(s) and s[left] == s[right]:
        count += 1
        left -= 1
        right += 1
    return count

# s = "aaa"
# print(palinSubStr(s))
##Time O(N^2)  space O(1)


##633. Sum of Square Numbers
# Input: c = 5
# Output: true
# Explanation: 1 * 1 + 2 * 2 = 5

def sumOfSquareNum(c):
    if c == 0:
        return True
    a = 0
    while a*a < c:
        b = c - a*a
        if perfectSqu(b):
            return True
        a += 1
    return False

def perfectSqu(num):
    left = 0
    right = num

    while left <= right:
        mid = left + (right-left)//2
        if mid*mid == num:
            return True
        if mid*mid < num:
            left = mid + 1
        else:
            right = mid - 1
    return False
# print(sumOfSquareNum(9))
##Time O(sqrt(c)logc)  space O(1)

##764. Largest Plus Sign

def largestPlusSignR(n, banned):
    res = 0
    table = [[0 for col in range(n)] for row in range(n)]
    mines = set()
    for d in banned:
        mines.add(tuple(d))

    for i in range(n):
        count = 0
        for j in range(n):
            if (i,j) in mines:
                count = 0
            else:
                count += 1
            table[i][j] = count
        count = 0
        for j in range(n-1, -1, -1):
            if (i,j) in mines:
                count = 0
            else:
                count += 1
            if count < table[i][j]:
                table[i][j] = count

    for j in range(n):
        count = 0
        for i in range(n):
            if (i,j) in mines:
                count = 0
            else:
                count += 1
            if count < table[i][j]:
                table[i][j] = count
        count = 0
        for i in range(n-1, -1, -1):
            if (i,j) in mines:
                count = 0
            else:
                count += 1
            if count < table[i][j]:
                table[i][j] = count
            if table[i][j] > res:
                res = table[i][j]
    return res

# n = 5
# banned = [[4,2]]
# print(largestPlusSignR(n, banned))
##Time O(N^2)   space O(N^2)

####765. Couples Holding Hands
##row = [0,2,1,3]
def couple(row):
    rowDict = {}
    for i,num in enumerate(row):
        rowDict[num] = i
    swaps = 0
    for i, num in enumerate(row):
        if i%2 == 0:
            if num%2 == 0:
                if row[i+1] != num + 1:
                    temIdx = rowDict[num+1]
                    row[i+1], row[temIdx] = row[temIdx], row[i+1]
                    rowDict[row[i+1]] = i + 1
                    rowDict[row[temIdx]] = temIdx
                    swaps += 1
            else:
                if row[i+1] != num - 1:
                    temIdx = rowDict[num - 1]
                    row[i+1], row[temIdx] = row[temIdx], row[i+1]
                    rowDict[row[i+1]] = i + 1
                    rowDict[row[temIdx]] = temIdx
                    swaps += 1
    return swaps

# row = [3,2,0,1]
# print(couple(row))
##Time O(N)  space O(N)

###767. Reorganize String
def reOrhanizeStr(s):
    dictS = {}
    for c in s:
        if c in dictS:
            dictS[c] += 1
        else:
            dictS[c] = 1

    res = ""
    maxHeap = []
    import heapq
    for c in dictS:
        heapq.heappush(maxHeap, (-dictS[c], c))
    while len(maxHeap) > 1:
        first = heapq.heappop(maxHeap)
        second = heapq.heappop(maxHeap)
        res += first[1]
        res += second[1]

        if first[0] + 1 != 0:
            heapq.heappush(maxHeap, (first[0]+1, first[1]))
        if second[0] + 1 != 0:
            heapq.heappush(maxHeap, (second[0]+1, second[1]))
    if len(maxHeap) != 0:
        leftElement = heapq.heappop(maxHeap)
        if leftElement[0] + 1 != 0:
            return ""
        else:
            return res + leftElement[1]

# s = "aaab"
# print(reOrhanizeStr(s))
##Time O(nLogn)  space O(N)

##785. Is Graph Bipartite?
##No two neighbour nodes will have same parent

def bipartite(graph):
    adjList = {}
    parent = {}
    for i in range(len(graph)):
        adjList[i] = []
        parent[i] = i
    # for i in range(len(graph)):
    #     neis = graph[i]
    #     for nei in neis:
    #         adjList[i].append(nei)
    #         adjList[nei].append(i)

    for node, neis in enumerate(graph):
        for nei in neis:
            if bipartiteFind(parent, node) == bipartiteFind(parent, nei):
                return False
            bipartiteUnion(parent, nei, neis[0])
    return True

def bipartiteFind(parent, x):
    if parent[x] != x:
        parent[x] = bipartiteFind(parent, parent[x])
    return parent[x]

def bipartiteUnion(parent, x, y):
    xP = bipartiteFind(parent, x)
    yP = bipartiteFind(parent, y)

    if xP != yP:
        parent[yP] = xP
        return True
    else:
        return False


# graph = [[1,3],[0,2],[1,3],[0,2]]
# print(bipartite(graph))
##Time O(ElogV)  space O(V)

###786. K-th Smallest Prime Fraction
# Input: arr = [1,2,3,5], k = 3
# Output: [2,5]
# Explanation: The fractions to be considered in sorted order are:
# 1/5, 1/3, 2/5, 1/2, 3/5, and 2/3.
# The third fraction is 2/5.

def kThSmallestPrimeFarc(arr, k):
    left = 0
    right = 1

    while left < right:
        mid = (left+right)/2
        count, p, q = primeFracCondition(arr, mid)
        if count == k:
            return [p, q]
        if count > k:
            right = mid
        else:
            left = mid


def primeFracCondition(arr, mid):
    maxVal = 0
    p = 0
    q = 0
    count = 0
    n = len(arr)
    j = 1
    for i in range(n-1):
        while j < n and arr[i]/arr[j] > mid:
            j += 1
        if j == n:
            break
        count += n - j
        if arr[i]/arr[j] > maxVal:
            maxVal = arr[i]/arr[j]
            p = arr[i]
            q = arr[j]
    return count, p, q

# arr = [1,7]
# k = 1
# print(kThSmallestPrimeFarc(arr, k))
##Time O(nlogn)  space O(1)

###791. Custom Sort String
def customSortStr(order, s):
    dictS = {}
    for c in s:
        if c in dictS:
            dictS[c] += 1
        else:
            dictS[c] = 1
    res = ""
    for c in order:
        if c in dictS:
            res += c*dictS[c]
            del dictS[c]

    if len(dictS) > 0:
        for c in dictS:
            res += c*dictS[c]
    return res

# order = "cbafg"
# s = "abcd"
# print(customSortStr(order, s))
##Time O(m+n)  space O(m)


###794. Valid Tic-Tac-Toe State

def validTictac(board):
    xCount = 0
    oCount = 0
    for i in range(3):
        for j in range(3):
            if board[i][j] == "X":
                xCount += 1
            if board[i][j] == "O":
                oCount += 1
    def win(board, player):
        for i in range(3):
            if board[i][0] == board[i][1] == board[i][2] == player:
                return True
        for j in range(3):
            if board[0][j] == board[1][j] == board[2][j] == player:
                return True
        if board[0][0] == board[1][1] == board[2][2] == player:
            return True
        if board[0][2] == board[1][1] == board[2][0] == player:
            return True
        return False

    if xCount not in [oCount, oCount+1]:
        return False
    if win(board, "X"):
        if xCount != oCount + 1:
            return False
    if win(board, "O"):
        if oCount != xCount:
            return False
    return True


# board = ["XOX","O O","XOX"]
# print(validTictac(board))
##Time O(1) space O(1)

##797. All Paths From Source to Target
# Input: graph = [[1,2],[3],[3],[]]
# Output: [[0,1,3],[0,2,3]]
# Explanation: There are two paths: 0 -> 1 -> 3 and 0 -> 2 -> 3.

def allPathsToTarget(graph):
    res = []
    n = len(graph)
    nodes = [i for i in range(len(graph))]
    stack = [(0, [0])]
    visited = set()
    while len(stack) != 0:
        node, path = stack.pop()
        #visited.add(node)
        if node == n - 1:
            res.append(path)
            continue
        else:
            for nei in graph[node]:
                #if nei not in visited:
                stack.append((nei, path+[nei]))
    return res

# graph = [[1,2],[3],[3],[]]
# print(allPathsToTarget(graph))
##Time O(VE)  space O(VE)


###824. Goat Latin
# Input: sentence = "I speak Goat Latin"
# Output: "Imaa peaksmaaa oatGmaaaa atinLmaaaaa"

def goatLatin(sentence):
    sent = sentence.split()
    for i, word in enumerate(sent):
        if word[0] in "aeiou" or word[0] in "AEIOU":
            sent[i] = word + "ma" + "a"*(i+1)
        else:
            first = word[0]
            newWord = word[1:] + first + "ma" + "a"*(i+1)
            sent[i] = newWord
    return " ".join(sent)

# sentence = "I speak Goat Latin"
# print(goatLatin(sentence))
##Time O(N)  space O(1)


###846. Hand of Straights
# Input: hand = [1,2,3,6,2,3,4,7,8], groupSize = 3
# Output: true
# Explanation: Alice's hand can be rearranged as [1,2,3],[2,3,4],[6,7,8]

def handOfStratight(hand, groupSize):
    if len(hand)%groupSize != 0:
        return False
    counter = {}
    for num in hand:
        if num in counter:
            counter[num] += 1
        else:
            counter[num] = 1
    import heapq
    minHeap = list(counter.keys())

    heapq.heapify(minHeap)
    while minHeap:
        first = minHeap[0]
        for i in range(first, first+groupSize):
            if i not in counter:
                return False
            counter[i] -= 1
            if counter[i] == 0:
                if i != minHeap[0]:
                    return False
                heapq.heappop(minHeap)
    return True

# hand = [1,2,3,4,5]
# groupSize = 4
# print(handOfStratight(hand, groupSize))
##Time O(nlogn)  space O(n)


###993. Cousins in Binary Tree
# Input: root = [1,2,3,4], x = 4, y = 3
# Output: false

def cousin(root, x , z):
    if root is None:
        return False
    if root.val == x or root.val == y:
        return False
    xInfo = []
    zInfo = []

    cousinDFS(root, None, x, xInfo, 0)
    cousinDFS(root, None, z, zInfo, 0)

    return xInfo[0][0] != yInfo[0][0] and xInfo[0][1] == yInfo[0][1]

def cousinDFS(root, parent, cousin, info, depth):
    if root is None:
        return None
    if root.val == cousin:
        info.append((parent.val, depth))

    cousinDFS(root.left, root, cousin, info, depth+1)
    cousinDFS(root.right, root, cousin, info, depth+1)


###980. Unique Paths III
# Input: grid = [[1,0,0,0],[0,0,0,0],[0,0,2,-1]]
# Output: 2

def uniquePath3(grid):
    if len(grid) == 0 or len(grid[0]) == 0:
        return 0
    zero = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 0:
                zero += 1
            elif grid[i][j] == 1:
                startX = i
                startY = j
    return uniquePath3DFS(grid, startX, startY, zero)


def uniquePath3DFS(grid, i, j, zero):
    if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] == -1:
        return 0
    if grid[i][j] == 2:
        if zero == -1:
            return 1
        else:
            return 0
    grid[i][j] = -1
    zero -= 1
    count = uniquePath3DFS(grid, i+1, j, zero) + uniquePath3DFS(grid, i-1, j, zero) + \
            uniquePath3DFS(grid, i, j+1, zero) + uniquePath3DFS(grid, i, j-1, zero)

    grid[i][j] = 0 ##BackTrack
    zero += 1
    return count

# grid = [[1,0,0,0],[0,0,0,0],[0,0,0,2]]
# print(uniquePath3(grid))
##Time O(3^n)  space O(n)

##1010. Pairs of Songs With Total Durations Divisible by 60
def pairDivBy60(time):
    dict1 = {}
    count = 0
    for i in range(len(time)):
        rem = time[i]%60
        if (60-rem)%60 in dict1:
            count += dict1[(60-rem)%60]
        if rem in dict1:
            dict1[rem] += 1
        else:
            dict1[rem] = 1
    return count

# time = [60,60,60]
# print(pairDivBy60(time))
##Time O(N)  space O(N)



##1011. Capacity To Ship Packages Within D Days
# Input: weights = [1,2,3,4,5,6,7,8,9,10], days = 5
# Output: 15

def minCapacityDdays(weights, days):
    left = min(weights)
    right = sum(weights)

    while left <= right:
        mid = left + (right-left)//2
        d = maxCapacityCondition(weights, mid)
        if d > days:
            left = mid + 1
        else:
            right = mid - 1
    return left

def maxCapacityCondition(weights, mid):
    days = 0
    cSum = 0
    for num in weights:
        if cSum + num > mid:
            days += 1
            cSum = 0
        cSum += num
    return days + 1

# weights = [3,2,2,4,1,4]
# days = 3
# print(minCapacityDdays(weights, days))
##Time O(nlogn)  space O(1)

##1013. Partition Array Into Three Parts With Equal Sum
def partitionArr3Equal(arr):
    if sum(arr) % 3 != 0:
        return False
    count = 0
    cSum = 0
    pSum = sum(arr)//3
    for num in arr:
        cSum += num
        if cSum == pSum:
            count += 1
            cSum = 0
    if count >= 3:
        return True
    else:
        return False

# arr = [3,3,6,5,-2,2,5,1,-9,4]
# print(partitionArr3Equal(arr))
##Time O(N)  space O(n)


###2124. Check if All A's Appears Before All B's
def checkIf(s):
    countOfA = 0
    for i in range(len(s)-1, -1, -1):
        if s[i] == "b":
            if countOfA > 0:
                return False
        else:
            countOfA += 1
    return True

# s = "bbb"
# print(checkIf(s))
##Time O(N)  space O(1)

##2064. Minimized Maximum of Products Distributed to Any Store
# Input: n = 6, quantities = [11,6]
# Output: 3

def prodDistribution(n, quantities):
    left = 1
    right = max(quantities)
    while left <= right:
        mid = left + (right-left)//2
        count = prodDictribution_condition(quantities, mid)
        if count > n:
            left = mid + 1
        else:
            right = mid - 1
    return left

def prodDictribution_condition(quantities, mid):
    count = 0
    for q in quantities:
        count += q // mid
        if q%mid != 0:
            count += 1
    return count

# n = 1
# quantities = [100000]
# print(prodDistribution(n, quantities))
##Time O(nlogn)  space O(1)


##1136. Parallel Courses
# Input: N = 3, relations = [[1,3],[2,3]]
# Output: 2
# Explanation:
# In the first semester, courses 1 and 2 are studied. In the second semester, course 3 is studied.
import collections
def parallelCourse(n, relations):
    graph = {}
    inDegree = {}

    for i in range(1, n+1):
        graph[i] = []
        inDegree[i] = 0
    for u, v in relations:
        graph[u].append(v)
        inDegree[v] += 1

    queue = collections.deque([])
    for i in range(1, n+1):
        if inDegree[i] == 0:
            queue.append(i)

    ans = 0
    totalCourses = 0
    while queue:
        ans += 1
        for _ in range(len(queue)):
            cur = queue.popleft()
            totalCourses += 1
            for nei in graph[cur]:
                inDegree[nei] -= 1
                if inDegree[nei] == 0:
                    queue.append(nei)
    if totalCourses == n:
        return ans
    else:
        return -1

# n = 3
# relations = [[1,2],[2,3],[3,1]]
# print(parallelCourse(n, relations))
##Time O(V+E)  space O(V+E)

##2050. Parallel Courses III
# Input: n = 3, relations = [[1,3],[2,3]], time = [3,2,5]
# Output: 8

def parallelCourse3(n, relations, time):
    graph = {}
    inDegree = {}
    for i in range(1, n+1):
        graph[i] = []
        inDegree[i] = 0

    for u, v in relations:
        graph[u].append(v)
        inDegree[v] += 1

    queue = collections.deque()
    dist = [0]*(n+1)
    for i in range(1, n+1):
        if inDegree[i] == 0:
            queue.append(i)
            dist[i] = time[i-1]

    while queue:
        for _ in range(len(queue)):
            cur = queue.popleft()
            for nei in graph[cur]:
                dist[nei] = max(dist[nei], dist[cur]+time[nei-1])
                inDegree[nei] -= 1
                if inDegree[nei] == 0:
                    queue.append(nei)
    return max(dist)

# n = 5
# relations = [[1,5],[2,5],[3,5],[3,4],[4,5]]
# time = [1,2,3,4,5]
# print(parallelCourse3(n, relations, time))
##Time O(v+E)  space O(V+E)


##2033. Minimum Operations to Make a Uni-Value Grid
# Input: grid = [[2,4],[6,8]], x = 2
# Output: 4
def minOperationUniGrid(grid, x):
    numList = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            numList.append(grid[i][j])
    numList.sort()
    n = len(numList)
    median = numList[n//2]

    steps = 0
    for num in numList:
        diff = abs(median - num)
        if diff%x != 0:
            return -1
        steps += diff//x
    return steps

# grid = [[1,2],[3,4]]
# x = 2
# print(minOperationUniGrid(grid, x))
##Time O(MN+nlogn) space O(m+n)

# def backTrack(i, cur, nums, ans):
#     if i == len(nums):
#         res = "".join(cur)
#         ans.append(res)
#         return
#     backTrack(i+1, cur, nums, ans)
#     cur[i] = "1"
#     backTrack(i+1, cur, nums, ans)
#
# nums = ["01","10"]
# cur = ["0" for c in nums]
# ans = []
# print(backTrack(0, cur, nums, ans))
# print(ans)

##1980. Find Unique Binary String
def uniqueBinaryStr(nums):
    res = ""
    for i in range(len(nums)):
        if nums[i][i] == "0":
            res += "1"
        else:
            res += "0"
    return res

# nums = ["111","011","001"]
# print(uniqueBinaryStr(nums))
##Time O(N)  space O(1)

##1976. Number of Ways to Arrive at Destination
# Input: n = 7, roads = [[0,6,7],[0,1,2],[1,2,3],[1,3,3],[6,3,3],[3,5,1],[6,5,1],[2,5,1],[0,4,5],[4,6,2]]
# Output: 4

import heapq
def numOfWays(n, roads):
    graph = {}
    for i in range(n):
        graph[i] = []
    for u,v,w in roads:
        graph[u].append((v, w))
        graph[v].append((u, w))

    dist = [float('inf')]*n
    dist[0] = 0
    ways = [0]*n
    ways[0] = 1
    pq = [(0,0)] ##Time, node
    while pq:
        oldDist, node = heapq.heappop(pq)
        for nei, d in graph[node]:
            newDist = oldDist + d
            if dist[nei] > newDist:
                dist[nei] = newDist
                ways[nei] = ways[node]
                heapq.heappush(pq, (newDist, nei))
            elif dist[nei] == newDist:
                ways[nei] += ways[node]
    return ways[n-1]

# n = 7
# roads = [[0,6,7],[0,1,2],[1,2,3],[1,3,3],[6,3,3],[3,5,1],[6,5,1],[2,5,1],[0,4,5],[4,6,2]]
# print(numOfWays(n, roads))
##Time O(V+ ElogV)  space O(V+E)

##1964. Find the Longest Valid Obstacle Course at Each Position
# Input: obstacles = [1,2,3,2]
# Output: [1,2,3,3]

def findLongestValid(obs):
    res = []
    arr = []
    for i in range(len(obs)):
        idx = bisectRight(arr, obs[i])
        if idx < len(arr):
            arr[idx] = obs[i]
        else:
            arr.append(obs[i])
        res.append(idx+1)
    return res

def bisectRight(nums, val):
    left = 0
    right = len(nums) - 1

    while left <= right:
        mid = left + (right-left)//2
        if val >= nums[mid]:
            left = mid + 1
        else:
            right = mid - 1
    return left

# obs = [3,1,5,6,4,2]
# print(findLongestValid(obs))
##Time O(nlogn)  space O(n)

###1793. Maximum Score of a Good Subarray
# Input: nums = [1,4,3,7,4,5], k = 3
# Output: 15
# Explanation: The optimal subarray is (1, 5) with a score of min(4,3,7,4,5) * (5-1+1) = 3 * 5 = 15.

def maxScoreGoodSubArray(nums, k):
    score = nums[k]
    left = k
    right = k
    res = nums[k]
    while left > 0 and right < len(nums) - 1:
        if nums[left-1] > nums[right+1]:
            left = left - 1
            score = min(nums[left], score)
        else:
            right += 1
            score = min(score, nums[right])
        res = max(res, score*(right-left+1))

    while left > 0:
        left -= 1
        score = min(score, nums[left])
        res = max(res, score * (right - left + 1))
    while right < len(nums) - 1:
        right += 1
        score = min(score, nums[right])
        res = max(res, score * (right - left + 1))
    return res

# nums = [5,5,4,5,4,1,1,1]
# k = 0
# print(maxScoreGoodSubArray(nums, k))
##Time O(N)  space O(1)

##1779. Find Nearest Point That Has the Same X or Y Coordinate
# Input: x = 3, y = 4, points = [[1,2],[3,1],[2,4],[2,3],[4,4]]
# Output: 2

def nearestPoint(x, y, points):
    d = float('inf')
    idx = -1
    for i, p in enumerate(points):
        x1 = p[0]
        y1 = p[1]
        if x == x1 or y == y1:
            dist = abs(x-x1) + abs(y-y1)
            if dist < d:
                d = dist
                idx = i
    return idx

# x = 3
# y = 4
# points = [[2,3]]
# print(nearestPoint(x, y, points))
##Time O(n)  space O(1)

###1760. Minimum Limit of Balls in a Bag
# Input: nums = [2,4,8,2], maxOperations = 4
# Output: 2

def minLimitOfBall(nums, maxOperations):
    left = 1
    right = max(nums)

    while left <= right:
        mid = left + (right-left)//2
        ops = limitOfBallCondition(nums, mid)
        if ops > maxOperations:
            left = mid + 1
        else:
            right = mid - 1
    return left

def limitOfBallCondition(nums, mid):
    ops = 0
    for num in nums:
        if num > mid:
            ops += (num-1)//mid
    return ops
# nums = [7,17]
# maxOperations = 2
# print(minLimitOfBall(nums, maxOperations))
##Time O(nlogn)  space O(1)


##1749. Maximum Absolute Sum of Any Subarray
# Input: nums = [1,-3,2,3,-4]
# Output: 5

def maxAbsoluteSum(nums):
    maxByEnd = nums[0]
    minByEnd = nums[0]
    maxSoFar = abs(nums[0])

    for i in range(1, len(nums)):
        maxByEnd = max(nums[i], maxByEnd+nums[i])
        minByEnd = min(nums[i], minByEnd+nums[i])

        maxSoFar = max(maxSoFar, abs(maxByEnd), abs(minByEnd))
    return maxSoFar

# nums = [2,-5,1,-4,3,-2]
# print(maxAbsoluteSum(nums))
##Time O(n) space O(1)

###1727. Largest Submatrix With Rearrangements
# Input: matrix = [[0,0,1],[1,1,1],[1,0,1]]
# Output: 4

def largestSubMatWithRearrange(matrix):
    for i in range(1, len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 1:
                matrix[i][j] += matrix[i-1][j]
            else:
                matrix[i][j] = 0

    res = 0
    for i in range(len(matrix)):
        sortedRow = sorted(matrix[i], reverse=True)
        for j in range(len(matrix[0])):
            res = max(res, sortedRow[j]*(j+1))
    return res

# matrix = [[1,1,0],[1,0,1]]
# print(largestSubMatWithRearrange(matrix))
##Time O(mnlogn)  space O(1)

##1721. Swapping Nodes in a Linked List
# Input: head = [1,2,3,4,5], k = 2
# Output: [1,4,3,2,5]

def swapNodes(head, k):
    if head is None:
        return None
    cur = head
    for _ in range(k):
        cur = cur.next

    node1 = cur
    node2 = head

    while cur.next:
        node2 = node2.next
        cur = cur.next
    node1.val, node2.val = node2.val, node1.val
    return head

##Time O(N)  space O(N)


##1658. Minimum Operations to Reduce X to Zero
# Input: nums = [1,1,4,2,3], x = 5
# Output: 2

def reduceXtoZero(nums, x):
    total = sum(nums)
    target = total - x
    if target == 0:
        return len(nums)
    elif target < 0:
        return -1

    cSum = 0
    left = 0
    ops = len(nums) + 1
    for i in range(len(nums)):
        cSum += nums[i]
        while left <= i and cSum > target:
            cSum -= nums[left]
            left += 1

        if cSum == target:
            ops = min(ops, len(nums)-(i-left+1))
    if ops == len(nums) + 1:
        return -1
    else:
        return ops

# nums = [3,2,20,1,1,3]
# x = 10
# print(reduceXtoZero(nums, x))
##Time O(N)  space O(1)

##1653. Minimum Deletions to Make String Balanced
# Input: s = "aababbab"
# Output: 2

def minDeletion(s):
    ops = 0
    numOfA = 0
    for i in range(len(s)-1, -1, -1):
        if s[i] == "b":
            if numOfA > 0:
                numOfA -= 1
                ops += 1
        else:
            numOfA += 1
    return ops

# s = "bbaaaaabb"
# print(minDeletion(s))
##Time O(N)  space O(1)

##1631. Path With Minimum Effort
# Input: heights = [[1,2,2],[3,8,2],[5,3,5]]
# Output: 2

def pathWIthMinEffort(heights):
    m = len(heights)
    n = len(heights[0])

    effort = [[float('inf') for col in range(n)] for row in range(m)]
    pq = [(0,0,0)] ##dist, row, col
    dirs = [(1,0), (0,1),(-1, 0),(0, -1)]
    while pq:
        val = heapq.heappop(pq)
        dist = val[0]
        row = val[1]
        col = val[2]
        if dist > effort[row][col]:
            continue
        if row == m -1 and col == n -1:
            return dist
        for d in dirs:
            newRow = row + d[0]
            newCol = col + d[1]
            if newRow >= 0 and newRow < m and newCol >= 0 and newCol < n:
                newDist = max(dist, abs(heights[newRow][newCol] - heights[row][col]))
                if newDist < effort[newRow][newCol]:
                    effort[newRow][newCol] = newDist
                    heapq.heappush(pq, (newDist, newRow, newCol))
    return 0

# heights = [[1,2,1,1,1],[1,2,1,2,1],[1,2,1,2,1],[1,2,1,2,1],[1,1,1,2,1]]
# print(pathWIthMinEffort(heights))
##Time O(mnlogmn)  space O(mn)


##705. Design HashSet
class myHashSet:
    ##Separate chainning to avoid collisin
    ###load factor = key/numOfBucket
    def __init__(self):
        self.numOfBucket = 15000
        self.bucket = [[] for i in range(self.numOfBucket)]

    def hashFunc(self, key):
        i = key%self.numOfBucket
        return i

    def add(self, key):
        i = self.hashFunc(key)
        if key not in self.bucket[i]:
            self.bucket[i].append(key)

    def remove(self, key):
        i = self.hashFunc(key)
        if key in self.bucket[i]:
            self.bucket[i].remove(key)

    def contain(self, key):
        i = self.hashFunc(key)
        if key in self.bucket[i]:
            return True
        return False

##706. Design HashMap
class hashMap:
    def __init__(self):
        self.map = [-1 for i in range(10**6+1)]

    def put(self, key, val):
        self.map[key] = val

    def get(self, key):
        return self.map[key]

    def remove(self, key):
        self.map[key] = -1


##712. Minimum ASCII Delete Sum for Two Strings
# Input: s1 = "sea", s2 = "eat"
# Output: 231

#
#   0 e a t
# 0 0 1 2 3
# s 1 1 3 4
# e 2 1 2 3
# a 3 2 1 2

def minAschII(s1, s2):
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)

    table = [[0 for col in range(len(s2)+1)] for row in range(len(s1)+1)]
    for i in range(len(table)):
        for j in range(len(table[0])):
            if i == 0:
                table[0][j] = sum(ord(c) for c in s2[:j])
            elif j == 0:
                table[i][0] = sum(ord(c) for c in s1[:i])
            else:
                if s1[i-1] == s2[j-1]:
                    table[i][j] = table[i-1][j-1]
                else:
                    table[i][j] = min(ord(s1[i-1])+table[i-1][j], ord(s2[j-1])+table[i][j-1])
    return table[len(s1)][len(s2)]

# s1 = "delete"
# s2 = "leet"
# print(minAschII(s1, s2))
##Time O(mn)  space O(mn)

##733. Flood Fill
# Input: image = [[1,1,1],[1,1,0],[1,0,1]], sr = 1, sc = 1, newColor = 2
# Output: [[2,2,2],[2,2,0],[2,0,1]]

def floodFill(image, sr, sc, newColor):
    oldColor = image[sr][sc]
    visited = [[False for col in range(len(image[0]))] for row in range(len(image))]
    floodFill_DFS(image, visited, sr, sc, oldColor, newColor)
    return image

def floodFill_DFS(image, visited, i, j, oldColor, newColor):
    if i < 0 or i >= len(image) or j < 0 or j >= len(image[0]) or image[i][j] != oldColor:
        return
    if visited[i][j]:
        return
    visited[i][j] = True
    image[i][j] = newColor
    floodFill_DFS(image, visited, i+1, j, oldColor, newColor)
    floodFill_DFS(image, visited, i-1, j, oldColor, newColor)
    floodFill_DFS(image, visited, i, j+1, oldColor, newColor)
    floodFill_DFS(image, visited, i, j-1, oldColor, newColor)

# image = [[0,0,0],[0,0,0]]
# sr = 0
# sc = 0
# newColor = 2
# print(floodFill(image, sr, sc, newColor))
##Time O(MN)  space O(MN)

###739. Daily Temperatures
# Input: temperatures = [73,74,75,71,69,72,76,73]
# Output: [1,1,4,2,1,1,0,0]

def dailyTemperature(temp):
    NG = [0]*len(temp)
    stack = []

    for i in range(len(temp)):
        while len(stack) != 0 and temp[stack[-1]] < temp[i]:
            NG[stack[-1]] = i
            stack.pop()
        stack.append(i)
    for i in range(len(temp)):
        if NG[i] != 0:
            NG[i] = NG[i] - i
    return NG

# temp = [30,60,90]
# print(dailyTemperature(temp))
##Time O(N) space O(N)

###743. Network Delay Time
# Input: times = [[2,1,1],[2,3,1],[3,4,1]], n = 4, k = 2
# Output: 2

def networkDelay(times, n, k):
    graph = {}
    for i in range(1, n+1):
        graph[i] = []
    for u, v, w in times:
        graph[u].append((v, w))

    dist = [float('inf')]*n
    dist[k-1] = 0
    pq = [(0, k)]  ## time, node

    while pq:
        w, u = heapq.heappop(pq)
        for nei, w1 in graph[u]:
            newWeight = w + w1
            if dist[nei-1] > newWeight:
                dist[nei-1] = newWeight
                heapq.heappush(pq, (newWeight, nei))

    maxW = 0
    for w in dist:
        maxW = max(maxW, w)
    if maxW == float('inf'):
        return -1
    else:
        return maxW

# times = [[1,2,1]]
# n = 2
# k = 2
# print(networkDelay(times, n, k))
##Time O(ElogV) space O(V+E)

###745. Prefix and Suffix Search
class trie:
    def __init__(self):
        self.children = {}
        self.maxIdx = -1
class wordFilter:
    def __init__(self, words):
        self.words = words
        self.root = trie()

        for i, word in enumerate(words):
             nWord = word + "#" + word
             for j in range(len(word)):
                 parent = self.root
                 for k in range(j, len(nWord)):
                     c = nWord[k]
                     if c not in parent.children:
                         parent.children[c] = trie()
                     parent = parent.children
                     parent.maxIdx = i

    def search(self, prefix, suffix):
        parent = self.root
        word = suffix + "#" + prefix
        for i, char in enumerate(word):
            if char not in parent:
                return -1
            parent = parent.children[char]
        return parent.maxIdex

##Time O(n*k^2 + Qk)  space O(n*k^2)

###746. Min Cost Climbing Stairs
# Input: cost = [10,15,20]
# Output: 15

def minCost(cost):
    if len(cost) <= 2:
        return min(cost)
    table = [0]*len(cost)
    n = len(table)
    table[0] = cost[0]
    table[1] = cost[1]
    for i in range(2, len(table)):
        table[i] = min(cost[i]+table[i-1], cost[i] + table[i-2])
    return min(table[n-1], table[n-2])

# cost = [1,100,1,1,1,100,1,1,100,1]
# print(minCost(cost))
##Time O(N)  space O(N)

##763. Partition Labels
# Input: s = "ababcbacadefegdehijhklij"
# Output: [9,7,8]

def partitionLabels(s):
    sDict = {}
    for i in range(len(s)):
        sDict[s[i]] = i
    start = 0
    end = 0
    res = []
    for i in range(len(s)):
        end = max(end, sDict[s[i]])
        if i == end:
            res.append(end-start+1)
            start = i + 1
    return res
# s = "eccbbbbdec"
# print(partitionLabels(s))
##Time O(N)  space O(N)

##764. Largest Plus Sign
# Input: n = 5, mines = [[4,2]]
# Output: 2

def largestPlusSign(n, mines):
    grid = [[0 for col in range(n)] for row in range(n)]
    banned = set()
    for m in mines:
        banned.add(tuple(m))
    ans = 0
    for i in range(n):
        count = 0
        for j in range(n):
            if (i,j) not in banned:
                count += 1
            else:
                count = 0
            grid[i][j] = count

        count = 0
        for j in range(n-1, -1, -1):
            if (i, j) not in banned:
                count += 1
            else:
                count = 0
            if grid[i][j] > count:
                grid[i][j] = count

    for j in range(n):
        count = 0
        for i in range(n):
            if (i,j) not in banned:
                count += 1
            else:
                count = 0
            if grid[i][j] > count:
                grid[i][j] = count

        count = 0
        for i in range(n-1, -1, -1):
            if (i,j) not in banned:
                count += 1
            else:
                count = 0
            if grid[i][j] > count:
                grid[i][j] = count
            ans = max(ans, grid[i][j])
    return ans

# n = 1
# mines = [[0,0]]
# print(largestPlusSign(n, mines))
##Time O(MN)  space O(MN)

##765. Couples Holding Hands
# Input: row = [0,2,1,3]
# Output: 1

def coupleHolding(row):
    rowDict = {}
    for i, num in enumerate(row):
        rowDict[num] = i
    swaps = 0
    for i, num in enumerate(row):
        if i%2 == 0:
            if num%2 == 0:
                if row[i+1] != num + 1:
                    tempIdx = rowDict[num+1]
                    row[i+1], row[tempIdx] = row[tempIdx], row[i+1]
                    rowDict[row[i+1]] = i + 1
                    rowDict[row[tempIdx]] = tempIdx
                    swaps += 1
            else:
                if row[i+1] != num - 1:
                    tempIdx = rowDict[num - 1]
                    row[i + 1], row[tempIdx] = row[tempIdx], row[i + 1]
                    rowDict[row[i + 1]] = i + 1
                    rowDict[row[tempIdx]] = tempIdx
                    swaps += 1
    return swaps

# row = [3,2,0,1]
# print(coupleHolding(row))
##Time O(N)  space O(N)

##767. Reorganize String
# Input: s = "aab"
# Output: "aba"

def reorganizeStr(s):
    sDict = {}
    for c in s:
        if c in sDict:
            sDict[c] += 1
        else:
            sDict[c] = 1
    res = ""
    maxHeap = []
    for key in sDict:
        heapq.heappush(maxHeap, (-sDict[key], key))

    while len(maxHeap) > 1:
        first = heapq.heappop(maxHeap)
        second = heapq.heappop(maxHeap)
        res += first[1]
        res += second[1]
        if first[0] + 1 != 0:
            heapq.heappush(maxHeap, (first[0]+1, first[1]))
        if second[0] + 1 != 0:
            heapq.heappush(maxHeap, (second[0]+1, second[1]))
    if len(maxHeap) != 0:
        last = heapq.heappop(maxHeap)
        if last[0] + 1 != 0:
            return ""
        res += last[1]
    return res

# s = "aaab"
# print(reorganizeStr(s))
##Time O(nlogn) space O(n)

##783. Minimum Distance Between BST Nodes
def minDist(root):
    if root is None:
        return None
    dist = float('inf')
    prev = None
    def inOrder(root):
        if root:
            if root.left:
                inOrder(root.left)
            if prev:
                dist = min(dist, root.val-prev.val)
            prev = root
            if root.right:
                inOrder(root.right)
    inOrder(root)
    return dist
##Time O(N)  space O(logN)

##791. Custom Sort String
# Input: order = "cba", s = "abcd"
# Output: "cbad"

def customSortStr(order, s):
    dictS = {}
    for c in s:
        if c in dictS:
            dictS[c] += 1
        else:
            dictS[c] = 1
    res = ""
    for char in order:
        if char in dictS:
            res += char*dictS[char]
            del dictS[char]
    for c in dictS:
        res += c*dictS[c]
    return res
# order = "cbafg"
# s = "abcd"
# print(customSortStr(order, s))
##Time O(m+n)  space O(m)

##794. Valid Tic-Tac-Toe State
def validTicTakToe(grid):
    if len(grid) == 0:
        return False
    xCount = 0
    oCount = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == "X":
                xCount += 1
            if grid[i][j] == "O":
                oCount += 1
    def win(grid, player):
        if grid[0][0] == grid[1][1] == grid[2][2] == player:
            return True
        if grid[0][2] == grid[1][1] == grid[2][0] == player:
            return True
        for i in range(len(grid)):
            if grid[i][0] == grid[i][1] == grid[i][2] == player:
                return True
        for j in range(len(grid[0])):
            if grid[0][j] == grid[1][j] == grid[2][j] == player:
                return True
        return False

    if oCount not in (xCount, xCount-1):
        return False
    if win(grid, "X"):
        if xCount != oCount + 1:
            return False
    if win(grid, "O"):
        if oCount != xCount:
            return False
    return True

# grid = ["XOX","O O","XOX"]
# print(validTicTakToe(grid))
##Time O(1)  space O(1)

##797. All Paths From Source to Target
# Input: graph = [[1,2],[3],[3],[]]
# Output: [[0,1,3],[0,2,3]]

def allPaths(graph):
    src = 0
    dst = len(graph) - 1
    stack = [(src, [src])]
    res = []
    while stack:
        node, path = stack.pop()
        if node == dst:
            res.append(path)
            continue
        else:
            for nei in graph[node]:
                stack.append((nei, path+[nei]))
    return res
# graph = [[4,3,1],[3,2,4],[3],[4],[]]
# print(allPaths(graph))
##Time O(V+E)  space O(V+E)

###815. Bus Routes
# Input: routes = [[1,2,7],[3,6,7]], source = 1, target = 6
# Output: 2
import collections
def busRoutes(routes, source, target):
    if source == target:
        return 0
    stopToBusId = {}
    for busId in range(len(routes)):
        for stop in routes[busId]:
            if stop in stopToBusId:
                stopToBusId[stop].append(busId)
            else:
                stopToBusId[stop] = [busId]

    stopSet = set()
    busSet = set()
    queue = collections.deque([source])
    ans = 0
    while queue:
        for _ in range(len(queue)):
            stop1 = queue.popleft()
            if stop1 == target:
                return ans
            stopSet.add(stop1)
            for bus_id in stopToBusId[stop1]:
                if bus_id not in busSet:
                    busSet.add(bus_id)
                    for newStop in routes[bus_id]:
                        if newStop not in stopSet:
                            queue.append(newStop)
        ans += 1
    return -1

# routes = [[7,12],[4,5,15],[6],[15,19],[9,12,13]]
# source = 15
# target = 12
# print(busRoutes(routes, source, target))
##Time O(MN)  space O(MN)

##846. Hand of Straights
# Input: hand = [1,2,3,6,2,3,4,7,8], groupSize = 3
# Output: true
def handOfS(hand, groupSize):
    if len(hand) % groupSize != 0:
        return False
    counter = {}
    for num in hand:
        if num in counter:
            counter[num] += 1
        else:
            counter[num] = 1
    minHeap = list(counter.keys())
    heapq.heapify(minHeap)

    while minHeap:
        first = minHeap[0]
        for i in range(first, first+groupSize):
            if i not in counter:
                return False
            counter[i] -= 1
            if counter[i] == 0:
                if i != minHeap[0]:
                    return False
                heapq.heappop(minHeap)
    return True
# hand = [1,2,3,6,2,3,4,7,8]
# groupSize = 3
# print(handOfS(hand, groupSize))
##Time O(nlogn)  space O(1)

##854. K-Similar Strings
# Input: s1 = "abc", s2 = "bca"
# Output: 2
def kSimilar(s1, s2):
    hight = {}
    hight[s1] = 0
    queue = collections.deque([s1])

    ans = 0
    while queue:
        for _ in range(len(queue)):
            s = queue.popleft()
            if s == s2:
                return hight[s]
            neis = getNeis(s, s2)
            for nei in neis:
                if nei not in hight:
                    hight[nei] = hight[s] + 1
                    queue.append(nei)
    return -1

def getNeis(s, s2):
    res = []
    i = 0
    while s[i] == s2[i]:
        i += 1
    sList = list(s)
    for j in range(i+1, len(sList)):
        if s[i] == s2[j]:
            sList[i], sList[j] = sList[j], sList[i]
            res.append("".join(sList))
            sList[i], sList[j] = sList[j], sList[i]
    return res
# s1 = "abc"
# s2 = "bca"
# print(kSimilar(s1, s2))
##Time O(n^2)  space O(N^2)

##875. Koko Eating Bananas
# Input: piles = [3,6,7,11], h = 8
# Output: 4

def koko(piles, h):
    left = 1
    right = max(piles)

    while left <= right:
        mid = left + (right-left)//2
        time = kokoCondition(piles, mid)
        if time > h:
            left = mid + 1
        else:
            right = mid - 1
    return left

def kokoCondition(piles, mid):
    time = 0
    for p in piles:
        time += p//mid
        if p%mid != 0:
            time += 1
    return time

# piles = [30,11,23,4,20]
# h = 6
# print(koko(piles, h))
##Time O(nlogn)  space O(1)

##896. Monotonic Array
# Input: nums = [1,2,2,3]
# Output: true

def monotonic(nums):
    increasing = True
    decreasing = True

    for i in range(1, len(nums)):
        if nums[i-1] > nums[i]:
            increasing = False
            break

    for i in range(1, len(nums)):
        if nums[i-1] < nums[i]:
            decreasing = False
            break
    if increasing or decreasing:
        return True
    else:
        return False


# nums = [1,3,2]
# print(monotonic(nums))
##Time O(N)  space O(N)

##897. Increasing Order Search Tree
class tree:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def increasingOrderSearchTree(root):
    if root is None:
        return None
    prev = None
    stack = []

    while root or stack:
        while root:
            stack.append(root)
            root = root.left
        node = stack.pop()
        if not prev:
            head = tree(node.val)
            head.left = None
            prev = head
        else:
            n = tree(node.val)
            prev.right = n
            n.left = None
            prev = n
        root = root.right
    return head

##Time O(N)  space O(N)


##901. Online Stock Span
class stockSpanner:
    def __init__(self):
        self.idx = -1
        self.stack = []
        self.res = 0

    def next(self, price):
        self.idx += 1
        while self.stack and price >= self.stack[-1][1]:
            self.stack.pop()
        if self.stack:
            self.res = self.idx - self.stack[-1][0]
        else:
            self.res = self.idx + 1
        self.stack.append((self.idx, price))
        return self.idx

##921. Minimum Add to Make Parentheses Valid
def minAdd(s):
    if len(s) == 0:
        return 0
    opening = 0
    count = 0
    for i in range(len(s)):
        if s[i] == "(":
            opening += 1
        else:
            if opening:
                opening -= 1
            else:
                count += 1
    return opening + count
# s = "((("
# print(minAdd(s))
##Time O(N) space O(1)


###334. Increasing Triplet Subsequence
# Input: nums = [1,2,3,4,5]
# Output: true
# Explanation: Any triplet where i < j < k is valid.

def triplet(nums):
    i = float('inf')
    j = float('inf')

    for k in range(len(nums)):
        if i > nums[k]:
            i = nums[k]
        elif j > nums[k]:
            j = nums[k]
        else:
            return True
    return False

# nums = [2,1,5,0,4,6]
# print(triplet(nums))
##Time O(N)  space O(1)


##337. House Robber III
def houseRob3(root):
    if root is None:
        return 0
    memo = {}
    res = houseRob3(root, memo)
    return res

def houseRob3(root, memo):
    if root is None:
        return 0
    if root in memo:
        return memo[root]
    value = root.val
    if root.left:
        value += houseRob3(root.left.left, memo) + houseRob3(root.left.right, memo)
    if root.right:
        value += houseRob3(root.right.left, memo) + houseRob3(root.right.right, memo)
    valWithoutRoot = houseRob3(root.left, memo) + houseRob3(root.right, memo)
    memo[root] = max(value, valWithoutRoot)
    return memo[root]

##Time O(N)  space O(logn)

##341. Flatten Nested List Iterator
class nestedIterator:
    def __init__(self, nestedList):
        self.nestedList = nestedList
        self.pointer = -1
        self.res = []
        self.iterator(nestedList)

    def iterator(self, nested):
        if nested is None:
            return
        for item in nested:
            if item.isInteger():
                self.res.append(item.getInteger())
            else:
                self.iterator( item.getList())

    def next(self):
        self.pointer += 1
        return self.res[self.pointer]

    def hasNext(self):
        return self.pointer < len(self.res) - 1

###345. Reverse Vowels of a String
# Input: s = "leetcode"
# Output: "leotcede"

def revVowel(s):
    s = list(s)
    left = 0
    right = len(s) - 1
    V = "aeiouAEIOU"
    while left <= right:
        if s[left] not in V and s[right] in V:
            left += 1
        elif s[left] in V and s[right] not in V:
            right -= 1
        elif s[left] in V and s[right] in V:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1

        else:
            left += 1
            right -= 1

    return "".join(s)

# s = "leetcode"
# print(revVowel(s))
##Time O(N)   space O(1)

###381. Insert Delete GetRandom O(1) - Duplicates allowed
class getRandomDuplicate:
    def __init__(self):
        self.array = []
        self.dict1 = collections.defaultdict(set)

    def insert(self, val):
        self.dict1[val].add(len(self.array))
        self.array.append(val)
        return len(self.dict1[val]) == 1

    def remove(self, val):
        if val not in self.dict1:
            return False
        idx = self.dict1[val].pop()
        lastElement = self.array[-1]
        self.array[idx] = lastElement
        lastIdx = len(self.array) - 1
        self.array.pop()
        self.dict1[lastElement].remove(lastIdx)
        self.dict1[lastElement].add(idx)
        return True

###394. Decode String
# Input: s = "3[a]2[bc]"
# Output: "aaabcbc"

def decodeStr(s):
    curStr = ""
    curCount = 0
    stack = []
    for i in range(len(s)):
        if s[i].isdigit():
            curCount = 10*curCount + int(s[i])
        elif s[i] == "[":
            stack.append(curStr)
            stack.append(curCount)
            curStr = ""
            curCount = 0

        elif s[i] == "]":
            count = stack.pop()
            prevStr = stack.pop()
            curStr = prevStr + curStr*count
        else:
            curStr += s[i]
    return curStr

# s = "2[abc]3[cd]ef"
# print(decodeStr(s))
##Time O(n)  space O(n)

##399. Evaluate Division
# Input: equations = [["a","b"],["b","c"]], values = [2.0,3.0], \
# queries = [["a","c"],["b","a"],["a","e"],["a","a"],["x","x"]]
# Output: [6.00000,0.50000,-1.00000,1.00000,-1.00000]

def evaluateDivision(equations, values, queries):
    graph = {}
    for i, (node1, node2) in enumerate(equations):
        if node1 not in graph:
            graph[node1] = []
        graph[node1].append((node2, values[i]))
        if node2 not in graph:
            graph[node2] = []
        graph[node2].append((node1, 1/values[i]))

    res = []
    for u, v in queries:
        seen = set()
        ans = evaluateDFS(graph, u, v, seen)
        res.append(ans)
    return res

def evaluateDFS(graph, u, v, seen):
    if u not in graph or v not in graph:
        return -1
    stack = [(u, 1)]
    while stack:
        cur = stack.pop()
        if cur[0] == v:
            return cur[1]
        if cur[0] not in seen:
            seen.add(cur[0])
            for nei, val in graph[cur[0]]:
                if nei not in seen:
                    stack.append((nei, val*cur[1]))
    return -1

# equations = [["a","b"],["b","c"]]
# values = [2.0,3.0]
# queries = [["a","c"],["b","a"],["a","e"],["a","a"],["x","x"]]
# print(evaluateDivision(equations, values, queries))
##Time O(Q*(V+E))  space O(V+E)

###403. Frog Jump
# Input: stones = [0,1,3,5,6,8,12,17]
# Output: true

def frogJump(stones):
    if len(stones) == 0:
        return True
    stoneDict = {}
    for stone in stones:
        stoneDict[stone] = set()
    stoneDict[0].add(0)

    for stone in stones:
        vals = stoneDict[stone]
        for k in vals:
            if stone + k - 1 in stoneDict:
                if k - 1 > 0:
                    stoneDict[stone+k-1].add(k-1)
            if stone + k in stoneDict:
                if k > 0:
                    stoneDict[stone+k].add(k)
            if stone + k + 1 in stoneDict:
                stoneDict[stone+k+1].add(k+1)
    if len(stoneDict[stones[-1]]) > 0:
        return True
    else:
        return False

# stones = [0,1,2,3,4,8,9,11]
# print(frogJump(stones))
##Time O(N^2)  space O(N)

##410. Split Array Largest Sum
# Input: nums = [7,2,5,10,8], m = 2
# Output: 18

def splitArrayLargest(nums, m):
    left = max(nums)
    right = sum(nums)

    while left <= right:
        mid = left + (right-left)//2
        count = splitArrayLargest_condition(nums, mid)
        if count > m:
            left = mid + 1
        else:
            right = mid - 1
    return left

def splitArrayLargest_condition(nums, mid):
    count = 1
    total = 0
    for num in nums:
        total += num
        if total > mid:
            total = num
            count += 1
    return count

# nums = [1,4,4]
# m = 3
# print(splitArrayLargest(nums, m))
##Time O(NlogN) space O(1)

###416. Partition Equal Subset Sum
# Input: nums = [1,5,11,5]
# Output: true
# Explanation: The array can be partitioned as [1, 5, 5] and [11].

def partitionEqualSubset(nums):
    if sum(nums) % 2 != 0:
        return False

    cSum = sum(nums) // 2
    memo = {}
    return partitionEqualSubset_helper(nums, cSum, 0, memo)

def partitionEqualSubset_helper(nums, cSum, i, memo):
    if cSum == 0:
        return True
    if cSum < 0 :
        return False
    if i >= len(nums):
        return False
    key = str(i) + "," + str(cSum)
    if key in memo:
        return memo[key]
    memo[key] = partitionEqualSubset_helper(nums, cSum-nums[i], i, memo) or partitionEqualSubset_helper(nums, cSum, i, memo)
    return memo[key]

# nums = [1,2,3,5]
# print(partitionEqualSubset(nums))
##Time O(N*cSum)

##429. N-ary Tree Level Order Traversal
def nArrayLevelOrder(root):
    if root is None:
        return []
    res = [[root.val]]
    stack = [root]

    while stack:
        cur = stack.pop()
        temp = []
        tempVal = []
        for node in stack:
            for n in node.children:
                temp.append(n)
                tempVal.append(n.val)
        if len(tempVal) != 0:
            res.append(tempVal)
        stack = temp
    return res

###438. Find All Anagrams in a String
# Input: s = "cbaebabacd", p = "abc"
# Output: [0,6]

def findAllAna(s, p):
    if len(p) == 0:
        return []
    res = []
    dictP = {}
    for c in p:
        if c in dictP:
            dictP[c] += 1
        else:
            dictP[c] = 1
    left = 0
    dictS = {}
    for i in range(len(s)):
        if s[i] in dictS:
            dictS[s[i]] += 1
        else:
            dictS[s[i]] = 1
        if i >= len(p):
            if dictS[s[i-len(p)]] == 1:
                del dictS[s[i-len(p)]]
            else:
                dictS[s[i-len(p)]] -= 1
        if dictS == dictP:
            res.append(i-len(p)+1)

    return res

# s = "cbaebabacd"
# p = "abc"
# print(findAllAna(s, p))
##Time O(N)  space O(N)

###442. Find All Duplicates in an Array
# Input: nums = [4,3,2,7,8,2,3,1]
# Output: [2,3]

def findAllDup(nums):
    duplicate = []
    for i in range(len(nums)):
        pos = abs(nums[i]) - 1
        if nums[pos] < 0:
            duplicate.append(abs(nums[i]))
        if nums[pos] > 0:
            nums[pos] *= -1
    return duplicate

# nums = [4,3,2,7,8,2,3,1]
# print(findAllDup(nums))
#Time O(n)

###453. Minimum Moves to Equal Array Elements
def minMove(num):
    Min = min(nums)
    steps = 0
    for num in nums:
        steps += num - Min
    return steps

# nums = [1,1,1]
# print(minMove(nums))
##Time O(N)  space O(1)

##462. Minimum Moves to Equal Array Elements II
def minMove2(nums):
    nums.sort()
    left = 0
    right = len(nums) - 1
    steps = 0
    while left < right:
        steps += nums[right] - nums[left]
        left += 1
        right -= 1
    return steps


##524. Longest Word in Dictionary through Deleting
# Input: s = "abpcplea", dictionary = ["ale","apple","monkey","plea"]
# Output: "apple"

def longestWordInDict(s, dictionary):
    if len(s) == 0 or len(dictionary) == 0:
        return ""
    res = ""
    for word in dictionary:
        if sequen(s, word):
            if len(res) < len(word) or (len(res) == len(word) and res > word):
                res = word
    return res

def sequen(s, word):
    if len(word) > len(s):
        return False
    i = 0
    j = 0
    while i < len(s) and j < len(word):
        if s[i] == word[j]:
            i += 1
            j += 1
        else:
            i += 1
    return j == len(word)

# s = "abpcplea"
# dictionary = ["a","b","c"]
# print(longestWordInDict(s, dictionary))
##Time O(MN) space O(1)  M == length of dictionary, N == length of s

###983. Minimum Cost For Tickets
# Input: days = [1,4,6,7,8,20], costs = [2,7,15]
# Output: 11

def minCostTickets(days, costs):
    maxDay = max(days)
    table = [0]*(maxDay + 1)
    days = set(days)
    res = 0
    for i in range(1, len(table)):
        if i in days:
            if i - 1 < 0:
                one = costs[0]
            else:
                one = table[i-1] + costs[0]

            if i - 7 < 0:
                seven = costs[1]
            else:
                seven = table[i-7] + costs[1]

            if i - 30 < 0:
                thirty = costs[2]
            else:
                thirty = table[i-30] + costs[2]
            table[i] = min(one, seven, thirty)
        else:
            table[i] = table[i-1]
    return table[-1]

# days = [1,4,6,7,8,20]
# costs = [2,7,15]
# print(minCostTickets(days, costs))
##Time O(N)  space O(N)

###1042. Flower Planting With No Adjacent
# Input: n = 3, paths = [[1,2],[2,3],[3,1]]
# Output: [1,2,3]

def planting(n, paths):
    graph = {}
    for i in range(1, n+1):
        graph[i] = []
    for u, v in paths:
        graph[u].append(v)
        graph[v].append(u)

    res = [0]*n
    for i in range(1, n+1):
        colors = {1,2,3,4}
        for nei in graph[i]:
            if res[nei-1] != 0 and res[nei-1] in colors:
                colors.remove(res[nei-1])
        res[i-1] = colors.pop()
    return res

##Time O(V+E)  space O(N)


###1048. Longest String Chain
# Input: words = ["a","b","ba","bca","bda","bdca"]
# Output: 4
# Explanation: One of the longest word chains is ["a","ba","bda","bdca"].

def longestChain(words):
    if len(words) == 0:
        return 0
    words = sorted(words, key=len)
    dict1 = {}
    res = 1
    for word in words:
        for i in range(len(word)):
            newStr = word[:i] + word[i+1:]
            if newStr in dict1:
                dict1[word] = dict1[newStr] + 1
                res = max(res, dict1[word])
        if word not in dict1:
            dict1[word] = 1
    return res

# words = ["xbc","pcxbcf","xb","cxbc","pcxbc"]
# print(longestChain(words))
##Time O(MN)  space O(N)

####1074. Number of Submatrices That Sum to Target
def subMatrixSumTarget(grid, target):
    if len(grid) == 0 or len(grid[0]) == 0:
        return 0
    for i in range(len(grid)):
        for j in range(1, len(grid)):
            grid[i][j] += grid[i][j-1]

    counter = 0
    for c1 in range(len(grid[0])):
        for c2 in range(c1, len(grid[0])):
            dict1 = {}
            dict1[0] = 1
            Sum = 0
            for row in range(len(grid)):
                if c1 > 0:
                    Sum += grid[row][c2] - grid[row][c1 -1]
                else:
                    Sum += grid[row][c2]

                if Sum - target in dict1:
                    counter += dict1[Sum - target]
                if Sum in dict1:
                    dict1[Sum] += 1
                else:
                    dict1[Sum] = 1
    return counter


# grid = [[904]]
# target = 0
# print(subMatrixSumTarget(grid, target))
##Time O(MN^2)  space O(M)

###1466. Reorder Routes to Make All Paths Lead to the City Zero
# Input: n = 6, connections = [[0,1],[1,3],[2,3],[4,0],[4,5]]
# Output: 3

def reOrder(n, connections):
    graph = {}
    conn = set()
    visited = set()
    for i in range(n):
        graph[i] = []
    for u, v in connections:
        graph[u].append(v)
        graph[v].append(u)
        conn.add((u, v))
    ans = reOrder_DFS(graph, 0, conn, visited)
    return ans

def reOrder_DFS(graph, node, conn, visited):
    res = 0
    if node in visited:
        return
    visited.add(node)
    for nei in graph[node]:
        if nei not in visited:
            if (node, nei) in conn:
                res += 1
            res += reOrder_DFS(graph, nei, conn, visited)
    return res

# n = 3
# connections = [[1,0],[2,0]]
# print(reOrder(n, connections))
##Time O(V+E)  space O(V+E)

###1472. Design Browser History
class node:
    def __init__(self, val):
        self.val = val
        self.next = None
        self.prev = None

class browserHistory:
    def __init__(self, homapage):
        self.head = node(homepage)

    def visit(self, url):
        newNode = node(url)
        self.head.next = newNode
        newNode.prev = self.head
        self.head = newNode

    def back(self, steps):
        while self.head.prev and steps:
            self.head = self.head.prev
            steps -= 1
        return self.head.val

    def forward(self, steps):
        while self.head.next and steps:
            self.head = self.head.next
            steps -= 1
        return self.head.val

####1489. Find Critical and Pseudo-Critical Edges in Minimum Spanning Tree
# Input: n = 5, edges = [[0,1,1],[1,2,1],[2,3,2],[0,3,2],[0,4,3],[3,4,3],[1,4,6]]
# Output: [[0,1],[2,3,4,5]]

class unionFind:
    def __init__(self, n):
        self.n = n
        self.parent = [i for i in range(n)]
        self.weight = 0
        self.e = 0

    def find(self, x):
        if x != self.parent[x]:
            self.parent[x] = self.find(self.parent[x])  ##Path compression
        return self.parent[x]

    def union(self, x, y, w):
        xP = self.find(x)
        yP = self.find(y)
        if xP != yP:
            self.parent[yP] = xP
            self.weight += w
            self.e += 1

def criticalSudoCritical(n, edges):
    Edges = [(w, u, v, i) for i, (u, v, w) in enumerate(edges)]
    Edges.sort()
    #minWeight = 0
    e1 = 0
    k = 0
    UF1 = unionFind(n)
    while e1 < n - 1:
        w, u, v, _ = Edges[k]
        k += 1
        if UF1.find(u) != UF1.find(v):
            UF1.union(u, v, w)
            e1 += 1
    minWeight = UF1.weight
    critical = []
    sudo = []
    ##Find critical
    for i in range(len(Edges)):
        UF2 = unionFind(n)
        for j in range(len(Edges)):
            if i == j:
                continue
            w, u, v, _ = Edges[j]
            if UF2.find(u) != UF2.find(v):
                UF2.union(u, v, w)
        if UF2.weight > minWeight or UF2.e < n - 1:
            critical.append(Edges[i][3])
        else:
            ##Check if i belong to any MST and force to include i and run MST algo again
            UF3 = unionFind(n)
            w, u, v, _ = Edges[i]
            UF3.union(u, v, w)
            for j in range(len(Edges)):
                if i == j:
                    continue
                w, u, v, _ = Edges[j]
                UF3.union(u, v, w)
            if UF3.weight == minWeight and UF3.e == n -1:
                sudo.append(Edges[i][3])
    return critical, sudo

# n = 4
# edges = [[0,1,1],[1,2,1],[2,3,1],[0,3,1]]
# print(criticalSudoCritical(n, edges))
##Time O(E^2)  space O(V+E)


###1537. Get the Maximum Score
# Input: nums1 = [2,4,5,8,10], nums2 = [4,6,8,9]
# Output: 30


def getMaxScore(nums1, nums2):
    s1 = 0
    s2 = 0
    res = 0
    i = 0
    j = 0

    while i < len(nums1) and j < len(nums2):
        if nums1[i] < nums2[j]:
            s1 += nums1[i]
            i += 1
        elif nums1[i] > nums2[j]:
            s2 += nums2[j]
            j += 1
        else:
            res += max(s1, s2) + nums1[i]
            s1 = 0
            s2 = 0
            i += 1
            j += 1

    while i < len(nums1):
        s1 += nums1[i]
        i += 1

    while j < len(nums2):
        s2 += nums2[j]
        j += 1

    res = (res + max(s1, s2))%(10**9 + 7)
    return res

# nums1 = [1,3,5,7,9]
# nums2 = [3,5,100]
# print(getMaxScore(nums1, nums2))
##Time O(N)  space O(1)