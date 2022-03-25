##23. Merge k Sorted Lists
class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None

def mergeKSorted(lists):
    if not lists or len(lists) == 0:
        return None
    amount = len(lists)
    interval = 1

    while interval < amount:
        for i in range(0, amount-interval, interval*2):
            lists[i] = merge(lists[i], lists[i+interval])
        interval *= 2
    return lists[0]


def merge(head1, head2):
    dummy = ListNode(0)
    temp = dummy

    while head1 and head2:
        if head1.val <= head2.val:
            temp.next = head1
            head1 = head1.next

        else:
            temp.next = head2
            head2 = head2.next
        temp = temp.next

    while head1:
        temp.next = head1
        temp = temp.next
        head1 = head1.next

    while head2:
        temp.next = head2
        temp = temp.next
        head2 = head2.next
    return dummy.next

##Time Complexity O(NlogK) --> N= Total number of nodes in two sorted lists K -> number of linked lists
##Space O(1)

##15. 3Sum

def threeSum(nums):
    nums.sort()
    if len(nums) < 3:
        return []
    res = []
    for i in range(len(nums)):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        left = i + 1
        right = len(nums) - 1

        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total == 0:
                res.append([nums[i], nums[left], nums[right]])
                while left <right and nums[left] == nums[left+1]:
                    left += 1
                while left < right and nums[right] == nums[right-1]:
                    right -= 1
                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1
    return res

# nums = [-1,0,1,2,-1,-4]
# print(threeSum(nums))
##Time complexity O(N^2) space O(N)

##238. Product of Array Except Self
def productExceptSelf(nums):
    res = []
    product = 1
    for num in nums:
        product *= num
        res.append(product)

    product = 1
    for i in range(len(nums)-1, 0, -1):
        res[i] = res[i-1]*product
        product *= nums[i]
    res[0] = product
    return res

# nums = [1,2,3,4]
# print(productExceptSelf(nums))
##Time complexity O(n) and space complexity O(1)


##17. Letter Combinations of a Phone Number
# Input: digits = "23"
# Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]

def letterCombination(digits):
    phoneDict = {"2":"abc", "3":"def", "4":"ghi","5":"jkl","6":"mno", "7":"pqrs",
                 "8":"tuv", "9":"wxyz"}
    res = []
    if len(digits) == 0:
        return res
    temp = []
    helper(phoneDict, res, 0, temp, digits)
    return res


def helper(phoneDict, res, i, temp, digits):
    if i == len(digits):
        res.append("".join(temp))
        return

    cur = phoneDict[digits[i]]
    for k in range(len(cur)):
        temp.append(cur[k])
        helper(phoneDict, res, i+1, temp, digits)
        temp.pop()

# digits = "23"
# print(letterCombination(digits))

##88. Merge Sorted Array
# Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
# Output: [1,2,2,3,5,6]

def mergeSortedArray(nums1, m, nums2, n):
    p1 = m - 1
    p2 = n - 1
    i = m + n - 1

    while p1 >= 0 and p2 >= 0:
        if nums1[p1] >= nums2[p2]:
            nums1[i] = nums1[p1]
            i -= 1
            p1 -= 1
        else:
            nums1[i] = nums2[p2]
            i -= 1
            p2 -= 1

    while p2 >= 0:
        nums1[i] = nums2[p2]
        p2 -= 1
        i -= 1
    return nums1

# nums1 = [1,2,3,0,0,0]
# m = 3
# nums2 = [2,5,6]
# n = 3
# print(mergeSortedArray(nums1, m, nums2, n))

##85. Maximal Rectangle

def maximalRectangle(grid):
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            grid[i][j] = int(grid[i][j])

    for i in range(1, len(grid)):
        for j in range(1, len(grid[0])):
            if grid[i][j] == 1:
                grid[i][j] += grid[i-1][j]

    maxArea = 0
    for row in grid:
        maxArea = max(maxArea, largestRectangle(row))
    return maxArea


def largestRectangle(heights):
    stack = []
    n = len(heights)
    left = [0]*n
    right = [0]*n

    for i in range(n):
        if len(stack) == 0:
            left[i] = 0
            stack.append(i)
        else:
            while len(stack) != 0 and heights[stack[-1]] >= heights[i]:
                stack.pop()
            if len(stack) == 0:
                left[i] = 0
            else:
                left[i] = stack[-1] + 1
            stack.append(i)

    stack = []
    for i in range(n-1, -1, -1):
        if len(stack) == 0:
            right[i] = n - 1
            stack.append(i)
        else:
            while len(stack) != 0 and heights[stack[-1]] >= heights[i]:
                stack.pop()
            if len(stack) == 0:
                right[i] = n - 1
            else:
                right[i] = stack[-1] - 1
            stack.append(i)
    maxArea = 0
    for i in range(n):
        maxArea = max(maxArea, (right[i]-left[i]+1)*heights[i])
    return maxArea

# grid = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
# print(maximalRectangle(grid))

###173. Binary Search Tree Iterator
class BSTIterator:
    def __init__(self, root):
        self.stack = []
        self.partialInorder(root)

    def partialInorder(self, root):
        while root:
            self.stack.append(root)
            root = root.left

    def next(self):
        top = self.stack.pop()
        self.partialInorder(top.right)
        return top.val

    def hasNext(self):
        return len(self.stack) != 0

##Time complexity O(1) in average space O(H)


###1586. Binary Search Tree Iterator II

class BST_iterator2:
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
            top = self.stack.pop()
            self.output.append(top)
            self.partialInorder(top.right)
            return top.val

    def prev(self):
        self.pointer -= 1
        res = self.output[self.pointer]
        return res.val

    def hasNext(self):
        if self.pointer + 1 >= 0 and self.pointer + 1 < len(self.output):
            return True
        return len(self.stack) != 0

    def hasPrev(self):
        if self.pointer - 1 >= 0 and self.pointer -1 < len(self.output):
            return True
        return False


##128. Longest Consecutive Sequence

# Input: nums = [100,4,200,1,3,2]
# Output: 4

def longestConSequence(nums):
    maxLen = 0
    curLen = 0
    if len(nums) == 0:
        return 0
    numSet = set(nums)

    for num in numSet:
        if num - 1 not in numSet:
            curNum = num
            curLen = 1

            while curNum + 1 in numSet:
                curLen += 1
                curNum = curNum + 1
            maxLen = max(maxLen, curLen)
    return maxLen

# nums = [0,3,7,2,5,8,4,6,0,1]
# print(longestConSequence(nums))
##Time O(n) and space O(1)


##341. Flatten Nested List Iterator
class nestedListIterator:
    def __init__(self, nestedList):
        self.pointer = -1
        self.res = []
        self.iterator(nestedList)

    def iterator(self, items):
        if items is None:
            return
        for item in items:
            if item.isInteger():
                self.res.append(item.getInteger())
            else:
                self.iterator(item.getList())

    def next(self):
        self.pointer += 1
        return self.res[self.pointer]

    def hasNext(self):
        return self.pointer < len(self.res) - 1


##301. Remove Invalid Parentheses
def removeInvalidParen(s):
    res = []
    strSet = set()
    count = invalidCount(s)
    solve(s, res, strSet, count)
    return res

def solve(s, res, strSet, count):
    if s in strSet:
        return
    else:
        strSet.add(s)
    if count < 0:
        return

    if count == 0:
        if invalidCount(s) == 0:
            res.append(s)
        return
    for i in range(len(s)):
        left = s[:i]
        right = s[i+1:]
        solve(left+right, res, strSet, count-1)


def invalidCount(s):
    stack = []
    for i in range(len(s)):
        if s[i] == "(":
            stack.append(s[i])
        elif s[i] == ")":
            if len(stack) != 0 and stack[-1] == "(":
                stack.pop()
            else:
                stack.append(s[i])
    return len(stack)

# s = ")("
# print(removeInvalidParen(s))

##76. Minimum Window Substring
# Input: s = "ADOBECODEBANC", t = "ABC"
# Output: "BANC"

def minWindowSubStr(s, t):
    if not s or not t:
        return ""
    dict_t = {}
    for char in t:
        if char in dict_t:
            dict_t[char] += 1
        else:
            dict_t[char] = 1

    filtered_s = []
    for i in range(len(s)):
        if s[i] in dict_t:
            filtered_s.append((i, s[i]))

    formed = 0
    required = len(dict_t)
    window_dict = {}
    res = [float('inf'), -1, -1]
    left = 0
    right = 0

    while right < len(filtered_s):
        char = filtered_s[right][1]
        window_dict[char] = window_dict.get(char, 0) + 1
        if window_dict[char] == dict_t[char]:
            formed += 1

        while left <= right and formed == required:
            start = filtered_s[left][0]
            end = filtered_s[right][0]
            if (end-start+1) < res[0]:
                res = [end-start+1, start, end]
            character = filtered_s[left][1]
            window_dict[character] -= 1
            if window_dict[character] < dict_t[character]:
                formed -= 1
            left += 1
        right += 1
    if res[0] == float('inf'):
        return ""
    else:
        return s[res[1] : res[2]+1]

# s = "a"
# t = "aa"
# print(minWindowSubStr(s, t))

##647. Palindromic Substrings
# Input: s = "abc"
# Output: 3
# Explanation: Three palindromic strings: "a", "b", "c".

def palindromicSubStr(s):
    total = 0
    for i in range(len(s)):
        total += palinCount(i, i, s)
        total += palinCount(i, i+1, s)
    return total


def palinCount(left, right, s):
    count = 0
    while left >= 0 and right < len(s) and s[left] == s[right]:
        count += 1
        left -= 1
        right += 1
    return count

# s = "aaa"
# print(palindromicSubStr(s))
##Time complexity O(n^2) space O(1)


##133. Clone Graph
class graphNode:
    def __init__(self, val = 0, neighbours= None):
        self.val = val
        self.neighbours = neighbours if neighbours is not None else []
def cloneGraph(node):
    clone = {}
    clone[node] = graphNode(node.val, [])
    import collections
    queue = collections.deque([])
    queue.append(node)

    while len(queue) != 0:
        cur = queue.popleft()
        for nei in cur.neighbours:
            if nei not in clone:
                clone[nei] = graphNode(nei.val, [])
                queue.append(nei)
            clone[cur].neighbours.append(clone[nei])
    return clone[node]

##Time O(VE)  space O(n)


##253. Meeting Rooms II
def meetingRoom2(intervals):
    intervals.sort(key=lambda x:x[0])
    room = 0
    import heapq
    minHeap = []
    for inter in intervals:
        start = inter[0]
        end = inter[1]
        if len(minHeap) != 0 and start >= minHeap[0]:
            heapq.heappop(minHeap)
        else:
            room += 1
        heapq.heappush(minHeap, end)
    return room

intervals = [[0, 5],[5, 10],[15, 20]]
print(meetingRoom2(intervals))


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

# nums = [2,1,5,0,4,6]
# print(increasingTriplet(nums))


###161. One Edit Distance

def oneEditDistance(s, t):
    if abs(len(s) - len(t)) > 1:
        return False
    i = 0
    j = 0
    count = 0
    while i < len(s) and j < len(t):
        if s[i] != t[j]:
            if count != 0:
                return False
            if len(s) > len(t):
                i += 1
            elif len(t) > len(s):
                j += 1
            else:
                i += 1
                j += 1
            count += 1
        i += 1
        j += 1
    if i < len(s) or j < len(s):
        count += 1
    return count == 1

# s = "1203"
# t = "1213"
# print(oneEditDistance(s, t))

##277. Find the Celebrity
def findCelebrity(graph):
    n = len(graph)
    celebrity = 0
    for i in range(1, n):
        if graph[celebrity][i] == 1:
            celebrity = i

    for i in range(n):
        if (i != celebrity & graph[celebrity][i]== 1) or graph[i][celebrity] == 0:
            return -1
    return celebrity

# graph = [
#   [1,0,1],
#   [1,1,0],
#   [0,1,1]
# ]
#
# print(findCelebrity(graph))

##



