##516. Longest Palindromic Subsequence

# Input: s = "bbbab"
# Output: 4
# Explanation: One possible longest palindromic subsequence is "bbbb".

def longPalSub(s):
    if len(s) < 2:
        return len(s)
    table = [[0 for col in range(len(s))] for row in range(len(s))]
    for i in range(len(s)):
        table[i][i] = 1
    for i in range(len(table)-1, -1, -1):
        for j in range(i+1, len(s)):
            if s[i] == s[j]:
                table[i][j] = 2 + table[i+1][j-1]
            else:
                table[i][j] = max(table[i+1][j], table[i][j-1])
    return table[0][-1]

# s = "bbbab"
# print(longPalSub(s))

##1161. Maximum Level Sum of a Binary Tree
def maxLevelSum(root):
    if root is None:
        return None
    minLevel = 1
    Sum = root.val
    stack = [root]
    dict1 = {}
    level = 1
    dict1[Sum] = 1
    while len(stack) != 0:
        tempLevel = []
        levelSum = 0
        size = len(stack)
        while size != 0:
            cur = stack.pop()
            if cur.left:
                tempLevel.append(cur.left)
                levelSum += cur.left.val
            if cur.right:
                tempLevel.append(cur.right)
                levelSum += cur.right.val
        stack = tempLevel
        level += 1
        dict1[levelSum] = level
        Sum = max(Sum, levelSum)
        minLevel = dict1[Sum]
    return minLevel

##1277. Count Square Submatrices with All Ones
# Input: matrix =
# [
#   [0,1,1,1],
#   [1,1,1,1],
#   [0,1,1,1]
# ]
# Output: 15
# Explanation:
# There are 10 squares of side 1.
# There are 4 squares of side 2.
# There is  1 square of side 3.
# Total number of squares = 10 + 4 + 1 = 15.


def squareWithAllOne(grid):
    table = [[0 for col in range(len(grid[0])+1)] for row in range(len(grid)+1)]

    count = 0
    for i in range(1, len(table)):
        for j in range(1, len(table[0])):
            if grid[i-1][j-1] == 1:
                table[i][j] = 1 + min(table[i-1][j], table[i][j-1], table[i-1][j-1])
                count += table[i][j]
    return count

# grid = [
#   [1,0,1],
#   [1,1,0],
#   [1,1,0]
# ]
#
# print(squareWithAllOne(grid))

##1283. Find the Smallest Divisor Given a Threshold
# Input: nums = [1,2,5,9], threshold = 6
# Output: 5
# Explanation: We can get a sum to 17 (1+2+5+9) if the divisor is 1.
# If the divisor is 4 we can get a sum of 7 (1+1+2+3) and if the divisor is 5 the sum will be 5 (1+1+1+2).

def smallestDivisor(nums, threshold):
    left = 1
    right = max(nums)

    while left <= right:
        mid = left + (right-left)//2
        value = condition(nums, threshold, mid)
        if value <= threshold:
            right = mid - 1
        else:
            left = mid + 1
    return left

def condition(nums, threshold, mid):
    value = 0
    for num in nums:
        value += num // mid
        if num%mid != 0:
            value += 1
    return value

# nums = [2,3,5,7,11]
# threshold = 11
# print(smallestDivisor(nums, threshold))


##410. Split Array Largest Sum
# Input: nums = [7,2,5,10,8], m = 2
# Output: 18
# Explanation:
# There are four ways to split nums into two subarrays.
# The best way is to split it into [7,2,5] and [10,8],
# where the largest sum among the two subarrays is only 18.

def splitArrayLargestSum(nums, m):
    left = max(nums)
    right = sum(nums)

    while left <= right:
        mid = left + (right-left)//2
        value = condition2(nums, mid)
        if value <= m:
            right = mid - 1
        else:
            left = mid + 1
    return left

def condition2(nums, mid):
    split = 0
    total = 0
    for num in nums:
        if total + num > mid:
            split += 1
            total = 0
        total += num
    return split + 1

# nums = [1,4,4]
# m = 3
# print(splitArrayLargestSum(nums, m))

##1296. Divide Array in Sets of K Consecutive Numbers
# Input: nums = [1,2,3,3,4,4,5,6], k = 4
# Output: true
# Explanation: Array can be divided into [1,2,3,4] and [3,4,5,6].

def kConsecutiveNums(nums, k):
    if len(nums) % k != 0:
        return False
    numDict = {}
    for num in nums:
        if num in numDict:
            numDict[num] += 1
        else:
            numDict[num] = 1

    import heapq
    minHeap = []
    keyList = list(numDict.keys())
    heapq.heapify(minHeap)

    while minHeap:
        first = minHeap[0]
        for i in range(first, first+k):
            if i not in numDict:
                return False
            numDict[i] -= 1
            if numDict[i] == 0:
                if i != minHeap[0]:
                    return False
                heapq.heappop(minHeap)
    return True

# nums = [1,2,3,4]
# k = 3
# print(kConsecutiveNums(nums, k))

##1304. Find N Unique Integers Sum up to Zero
# Input: n = 5
# Output: [-7,-1,1,3,4]
# Explanation: These arrays also are accepted [-5,-1,1,2,3] , [-3,-1,2,-2,4].

def sumUpToZero(n):
    res = []
    for i in range(1, n//2+1):
        res.append(i)
        res.append(-i)
    if n%2 !=0:
        res.append(0)
    return res

# print(sumUpToZero(10))

#1458. Max Dot Product of Two Subsequences
#    3 0 -6
#
#  2 6 6 6
#  1 6 6
# -2 6
#  5 15
def maxDotProd(nums1, nums2):
    table = [[0 for col in range(len(nums2))] for row in range(len(nums1))]
    table[0][0] = nums1[0]*nums2[0]
    for i in range(1, len(table)):
        table[i][0] = max(table[i-1][0], nums1[i]*nums2[0])
    for j in range(1, len(table[0])):
        table[0][j] = max(table[0][j-1], nums1[0]*nums2[j])

    for i in range(1, len(table)):
        for j in range(1, len(table[0])):
            table[i][j] = max(table[i-1][j], table[i][j-1], table[i-1][j-1]+nums1[i]*nums2[j], nums1[i]*nums2[j])
    return table[-1][-1]

# nums1 = [2,1,-2,5]
# nums2 = [3,0,-6]
# print(maxDotProd(nums1, nums2))

##1480. Running Sum of 1d Array
# Input: nums = [1,2,3,4]
# Output: [1,3,6,10]
# Explanation: Running sum is obtained as follows: [1, 1+2, 1+2+3, 1+2+3+4].

def runningSum(nums):
    for i in range(1, len(nums)):
        nums[i] += nums[i-1]
    return nums
# nums = [1,2,3,4]
# print(runningSum(nums))

##1509. Minimum Difference Between Largest and Smallest Value in Three Moves
# Input: nums = [5,3,2,4]
# Output: 0
# Explanation: Change the array [5,3,2,4] to [2,2,2,2].
# The difference between the maximum and minimum is 2-2 = 0.

def minDiff(nums):
    if len(nums) <= 4:
        return 0
    nums.sort()
    n = len(nums)
    op1 = nums[n-4] - nums[0]
    op2 = nums[n-3] - nums[1]
    op3 = nums[n-2] - nums[2]
    op4 = nums[n-1] - nums[3]
    return min(op1, op2, op3, op4)

# nums = [6,6,0,1,1,4,6]
# print(minDiff(nums))

##1525. Number of Good Ways to Split a String
# Input: s = "aacaba"
# Output: 2
# Explanation: There are 5 ways to split "aacaba" and 2 of them are good.

def numSplits(s):
    prefix = [0]*len(s)
    prefixSet = set()

    suffix = [0]*len(s)
    suffixSet = set()

    for i in range(len(s)):
        prefixSet.add(s[i])
        prefix[i] = len(prefixSet)

    for i in range(len(s)-1, -1, -1):
        suffixSet.add(s[i])
        suffix[i] = len(suffixSet)

    count = 0
    for i in range(1, len(suffix)):
        if prefix[i-1] == suffix[i]:
            count += 1
    return count

# s = "acbadbaada"
# print(numSplits(s))

##1537. Get the Maximum Score
# Input: nums1 = [2,4,5,8,10], nums2 = [4,6,8,9]
# Output: 30
# Explanation: Valid paths:
# [2,4,5,8,10], [2,4,5,8,9], [2,4,6,8,9], [2,4,6,8,10],  (starting from nums1)
# [4,6,8,9], [4,5,8,10], [4,5,8,9], [4,6,8,10]    (starting from nums2)
# The maximum is obtained with the path in green [2,4,6,8,10].

def maxScore(nums1, nums2):
    i = 0
    j = 0
    s1 = 0
    s2 = 0
    res = 0

    while i < len(nums1) and j < len(nums2):
        if nums1[i] < nums2[j]:
            s1 += nums1[i]
            i += 1
        elif nums1[i] > nums2[j]:
            s2 += nums2[j]
            j += 1
        else:
            res += max(s1, s2) + nums1[i]
            i += 1
            j += 1
            s1 = 0
            s2 = 0

    while i < len(nums1):
        s1 += nums1[i]
        i += 1

    while j < len(nums2):
        s2 += nums2[j]
        j += 1

    return max(s1, s2) + res

# nums1 = [1,4,5,8,9,11,19]
# nums2 = [2,3,4,11,12]
# print(maxScore(nums1, nums2))


##1631. Path With Minimum Effort
# Input: heights = [[1,2,2],[3,8,2],[5,3,5]]
# Output: 2
# Explanation: The route of [1,3,5,3,5] has a maximum absolute difference of 2 in consecutive cells.
# This is better than the route of [1,2,2,2,5], where the maximum absolute difference is 3.

def minEffortPath(heights):
    m = len(heights)
    n = len(heights[0])
    import heapq
    table = [[float('inf') for col in range(n)] for row in range(m)]

    heap = [(0,0,0)]
    directions = [(1,0),(-1,0),(0,1),(0,-1)]
    while len(heap) != 0:
        dist, row, col = heapq.heappop(heap)
        if dist > table[row][col]:
            continue
        if row == m-1 and col == n-1:
            return dist
        for d in directions:
            newRow = row + d[0]
            newCol = col + d[1]
            if newRow >= 0 and newRow < m and newCol >= 0 and newCol < n:
                newDist = max(dist, abs(heights[row][col]-heights[newRow][newCol]))
                if newDist < table[newRow][newCol]:
                    table[newRow][newCol] = newDist
                    heapq.heappush(heap, (newDist, newRow, newCol))
    return 0

# heights = [[1,2,1,1,1],[1,2,1,2,1],[1,2,1,2,1],[1,2,1,2,1],[1,1,1,2,1]]
# print(minEffortPath(heights))

##1656. Design an Ordered Stream
class orderStream:
    def __init__(self, n):
        self.n = n
        self.pointer = 0
        self.res = [None]*n
    def insert(self, key, value):
        ans = []
        self.res[key-1] = value

        while self.pointer < len(self.res) and self.res[self.pointer] != None:
            ans.append(self.res[self.pointer])
            self.pointer += 1
        return ans

# os = orderStream(5)
# print(os.insert(3, "ccccc"))
# print(os.insert(1, "aaaaa"))
# print(os.insert(2, "bbbbb"))
# print(os.insert(5, "eeeee"))
# print(os.insert(4, "ddddd"))

##1658. Minimum Operations to Reduce X to Zero
# Input: nums = [1,1,4,2,3], x = 5
# Output: 2
# Explanation: The optimal solution is to remove the last two elements to reduce x to zero.

def minOperation(nums, x):
    totalSum = sum(nums)
    target = totalSum - x
    if target == 0:
        return len(nums)
    if target < 0:
        return -1
    cSum = 0
    left = 0
    ops =  len(nums) + 1
    for i in range(len(nums)):
        cSum += nums[i]
        while cSum > target and left <= i:
            cSum -= nums[left]
            left += 1
        if cSum == target:
            ops = min(ops, len(nums) - (i-left+1))
    if ops == len(nums) + 1:
        return -1
    else:
        return ops

# nums = [3,2,20,1,1,3]
# x = 10
# print(minOperation(nums, x))

##692. Top K Frequent Words
# Input: words = ["i","love","leetcode","i","love","coding"], k = 2
# Output: ["i","love"]
# Explanation: "i" and "love" are the two most frequent words.
# Note that "i" comes before "love" due to a lower alphabetical order.

def topKFrequent(words, k):
    import heapq
    wordDict = {}
    for word in words:
        if word in wordDict:
            wordDict[word] += 1
        else:
            wordDict[word] = 1
    heap = []
    for key, value in wordDict.items():
        heapq.heappush(heap, (-value, key))

    res = []
    for i in range(k):
        res.append(heapq.heappop(heap)[1])
    return res

##Time O(nlogn) space O(n)
# words = ["the","day","is","sunny","the","the","the","sunny","is","is"]
# k = 4
# print(topKFrequent(words, k))

##695. Max Area of Island
def maxAreaOfIsland(grid):
    m = len(grid)
    n = len(grid[0])

    maxArea = 0

    visited = [[False for col in range(n)] for row in range(m)]

    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                area = island_DFS(grid, i, j, visited)
                maxArea = max(maxArea, area)
    return maxArea

def island_DFS(grid, i, j, visited):
    if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] == 0:
        return 0
    if visited[i][j]:
        return 0
    visited[i][j] = True
    area = 1
    area += island_DFS(grid, i + 1, j, visited)
    area += island_DFS(grid, i - 1, j, visited)
    area += island_DFS(grid, i, j+1, visited)
    area += island_DFS(grid, i, j-1, visited)
    return area
#Time O(mn) space O(mn)
# grid = [[0,0,0,0,0,0,0,0]]
# print(maxAreaOfIsland(grid))

##697. Degree of an Array
# Input: nums = [1,2,2,3,1]
# Output: 2
# Explanation:
# The input array has a degree of 2 because both elements 1 and 2 appear twice.
# Of the subarrays that have the same degree:
# [1, 2, 2, 3, 1], [1, 2, 2, 3], [2, 2, 3, 1], [1, 2, 2], [2, 2, 3], [2, 2]
# The shortest length is 2. So return 2.

def degreeOfArray(nums):
    left = {} # left index of num
    right = {}  # right index of num
    counter = {} # counter of num

    for i in range(len(nums)):
        if nums[i] not in left:
            left[nums[i]] = i
        right[nums[i]] = i
        counter[nums[i]] = counter.get(nums[i], 0) + 1
    degree = max(counter.values())
    ans = len(nums)
    for num in nums:
        if counter[num] == degree:
            ans = min(ans, (right[num]-left[num]+1))
    return ans

##Time O(n) and space O(n)
# nums = [1,2,2,3,1,4,2]
# print(degreeOfArray(nums))

##700. Search in a Binary Search Tree
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def searchInBST(root, val):
    if root is None:
        return None
    if root.val == val:
        return root

    if root.val > val:
        return searchInBST(root.left, val)
    else:
        return searchInBST(root.right, val)

##701. Insert into a Binary Search Tree

def insertInBST(root, val):
    if root is None:
        root = TreeNode(val)
        return root
    _insert(root, val)
    return root

def _insert(root, val):
    if root.val > val:
        if root.left:
            _insert(root.left, val)
        else:
            root.left = TreeNode(val)
    if root.val < val:
        if root.right:
            _insert(root.right, val)
        else:
            root.right = TreeNode(val)


##703. Kth Largest Element in a Stream
import heapq
class kThLargest:
    def __init__(self, k, nums):
        self.k = k
        self.nums = nums
        heapq.heapify(self.nums)

        while len(self.nums) > k:
            heapq.heappop(self.nums)

    def add(self, val):
        heapq.heappush(self.nums, val)
        if len(self.nums) > self.k:
            heapq.heappop(self.nums)
        return self.nums[0]

# nums = [4, 5, 8, 2]
# k = 3
# k_largest = kThLargest(k, nums)
# print(k_largest.add(3))
# print(k_largest.add(5))
# print(k_largest.add(10))
# print(k_largest.add(9))
# print(k_largest.add(4))

##1721. Swapping Nodes in a Linked List
def swapNodes(head, k):
    cur = head

    while k-1 != 0:
        cur = cur.next
    node1 = cur
    cur = head

    while node1.next:
        cur = cur.next
        node1 = node1.next
    node2 = cur
    node1.val, node2.val = node2.val, node1.val
    return head

##19. Remove Nth Node From End of List
def removenTh(head, n):
    slow = head
    fast = head

    for i in range(n):
        fast = fast.next

    if fast is None:
        return head.next

    while fast.next != None:
        fast = fast.next
        slow = slow.next
    slow.next = slow.next.next
    return head

##24. Swap Nodes in Pairs
class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None

def swapNodes(head):
    cur1 = ListNode(0)
    cur = cur1

    while cur.next and cur.next.next:
        p = cur.next
        q = cur.next.next

        cur.next = q
        p.next = q.next
        q.next = p

        cur = p
    return cur1.next


##1727. Largest Submatrix With Rearrangements
# Input: matrix = [[0,0,1],[1,1,1],[1,0,1]]
# Output: 4
# Explanation: You can rearrange the columns as shown above.
# The largest submatrix of 1s, in bold, has an area of 4.

def largestSubMatrix(grid):
    m = len(grid)
    n = len(grid[0])

    for j in range(n):
        for i in range(m-2, -1, -1):
            if grid[i][j]:
                grid[i][j] += grid[i+1][j]
    print(grid)

    ans = 0
    for i in range(m):
        sortedRow = sorted(grid[i], reverse=True)
        for j in range(n):
            ans = max(ans, sortedRow[j]*(j+1))
    return ans


# grid = [[0,0],[0,0]]
# print(largestSubMatrix(grid))

##1749. Maximum Absolute Sum of Any Subarray
def maxAbsoluteSum(nums):
    maxByend = nums[0]
    minByEnd = nums[0]

    maxSoFar = nums[0]

    for i in range(1, len(nums)):
        maxByend = max(nums[i], maxByend+nums[i])
        minByEnd = min(nums[i], minByEnd+nums[i])

        maxSoFar = max(abs(maxByend), abs(minByEnd), maxSoFar)

    return maxSoFar

# nums = [2,-5,1,-4,3,-2]
# print(maxAbsoluteSum(nums))

##1760. Minimum Limit of Balls in a Bag
# Input: nums = [2,4,8,2], maxOperations = 4
# Output: 2

def minLimit(nums, maxOperations):
    left = 1
    right = max(nums)
    while left <= right:
        mid = left + (right-left)//2
        operations = minLimit_condition(nums, mid)
        if operations <= maxOperations:
            right = mid - 1
        else:
            left = mid + 1
    return left

def minLimit_condition(nums, mid):
    operations = 0
    for num in nums:
        if num > mid:
            operations += (num-1) // mid
    return operations

# nums = [7,17]
# maxOperations = 2
# print(minLimit(nums, maxOperations))


##1779. Find Nearest Point That Has the Same X or Y Coordinate
# Input: x = 3, y = 4, points = [[1,2],[3,1],[2,4],[2,3],[4,4]]
# Output: 2
# Explanation: Of all the points, only [3,1], [2,4] and [4,4] are valid. Of the valid points, [2,4] and [4,4]
# have the smallest Manhattan distance from your current location, with a distance of 1. [2,4] has the smallest index,
# so return 2.

def nearestPoint(x, y, points):
    idx = -1
    dist = float('inf')
    for i, point in enumerate(points):
        x1 = point[0]
        y1 = point[1]
        if x == x1 or y == y1:
            dist1 = abs(x-x1) + abs(y-y1)
            if dist > dist1:
                dist = dist1
                idx = i
    return idx

# x = 3
# y = 4
# points = [[1,2],[3,1],[2,4],[2,3],[4,4]]
# print(nearestPoint(x, y, points))

##973. K Closest Points to Origin
# Input: points = [[3,3],[5,-1],[-2,4]], k = 2
# Output: [[3,3],[-2,4]]
# Explanation: The answer [[-2,4],[3,3]] would also be accepted.

def kClosestPoints(points, k):
    import math
    d_list = []
    for point in points:
        dist = math.sqrt(point[0]**2 + point[1]**2)
        d_list.append((dist, point))

    left = 0
    right = len(points) - 1

    while left <= right:
        idx = partition(left, right, d_list)
        if idx == k - 1:
            break
        if idx < k -1:
            left = idx + 1
        else:
            right = idx - 1

    res = []
    for i in d_list[:idx+1]:
        res.append(i[1])
    return res


def partition(first_index, last_index, nums):
    pivot = (first_index+last_index) // 2
    nums[last_index], nums[pivot] = nums[pivot], nums[last_index]
    for i in range(first_index, last_index):
        if nums[i][0] < nums[last_index][0]:
            nums[i], nums[first_index] = nums[first_index], nums[i]
            first_index += 1
    nums[first_index], nums[last_index] = nums[last_index], nums[first_index]
    return first_index

# points = [[3,3],[5,-1],[-2,4]]
# k = 2
# print(kClosestPoints(points, k))

##1793. Maximum Score of a Good Subarray
# Input: nums = [1,4,3,7,4,5], k = 3
# Output: 15
# Explanation: The optimal subarray is (1, 5) with a score of min(4,3,7,4,5) * (5-1+1) = 3 * 5 = 15.

def maxScore(nums, k):
    if len(nums) == 0:
        return 0
    left = right = k
    minVal = nums[k]
    res = nums[k]

    while left > 0 or right < len(nums) - 1:
        if left > 0 and right < len(nums) - 1:
            if nums[left-1] > nums[right+1]:
                left -= 1
                minVal = min(minVal, nums[left])
            else:
                right += 1
                minVal = min(minVal, nums[right])

        elif left > 0:
            left -= 1
            minVal = min(minVal, nums[left])
        else:
            right += 1
            minVal = min(minVal, nums[right])
        res = max(res, minVal*(right-left+1))
    return res

#Time O(n) and space O(1)
# nums = [5,5,4,5,4,1,1,1]
# k = 0
# print(maxScore(nums, k))

##1886. Determine Whether Matrix Can Be Obtained By Rotation
# Input: mat = [[0,1],[1,0]], target = [[1,0],[0,1]]
# Output: true
# Explanation: We can rotate mat 90 degrees clockwise to make mat equal target.

def obtainByRotation(mat, target):
    rotate = 0
    while rotate <= 3:
        mat = rotate90(mat)
        if mat == target:
            return True
        rotate += 1
    return False


def rotate90(mat):
    ##transpose the matrix
    for i in range(len(mat)):
        for j in range(i, len(mat[0])):
            mat[i][j], mat[j][i] = mat[j][i], mat[i][j]
    ##Reverse rows
    for i in range(len(mat)):
        mat[i] = mat[i][::-1]

    return mat

# mat = [[0,0,0],[0,1,0],[1,1,1]]
# target = [[1,1,1],[0,1,0],[0,0,0]]
# print(obtainByRotation(mat, target))

##1945. Sum of Digits of String After Convert
# Input: s = "iiii", k = 1
# Output: 36
# Explanation: The operations are as follows:
# - Convert: "iiii" ➝ "(9)(9)(9)(9)" ➝ "9999" ➝ 9999
# - Transform #1: 9999 ➝ 9 + 9 + 9 + 9 ➝ 36
# Thus the resulting integer is 36.

def afterConvert(s, k):
    alphas = "abcdefghijklmnopqrstuvwxyz"
    alphaDict = {}
    val = 1
    for c in alphas:
        alphaDict[c] = val
        val += 1
    res = ""
    for c in s:
        res += str(alphaDict[c])

    for _ in range(k):
        Sum = 0
        for c in res:
            Sum += int(c)
        print(Sum)
        res = str(Sum)
    return int(res)

#
# s = "leetcode"
# k = 2
# print(afterConvert(s, k))

##1980. Find Unique Binary String
# Input: nums = ["01","10"]
# Output: "11"
# Explanation: "11" does not appear in nums. "00" would also be correct.

##Cantor's diagonal argument method

def uniqueBinaryString(nums):
    res = ""
    for i in range(len(nums)):
        if nums[i][i] == "0":
            res += "1"
        else:
            res += "0"
    return res
##Time O(n) and space O(1)
# nums = ["111","011","001"]
# print(uniqueBinaryString(nums))

##1985. Find the Kth Largest Integer in the Array
# Input: nums = ["3","6","7","10"], k = 4
# Output: "3"
# Explanation:
# The numbers in nums sorted in non-decreasing order are ["3","6","7","10"].
# The 4th largest integer in nums is "3".

def kthLargest(nums, k):
    import heapq
    maxHeap = [-int(x) for x in nums]
    heapq.heapify(maxHeap)

    while k > 1:
        heapq.heappop(maxHeap)
        k -= 1
    res = heapq.heappop(maxHeap)
    return str(-1*res)

# nums = ["0","0"]
# k = 2
# print(kthLargest(nums, k))

##1991. Find the Middle Index in Array
# Input: nums = [2,3,-1,8,4]
# Output: 3
# Explanation:
# The sum of the numbers before index 3 is: 2 + 3 + -1 = 4
# The sum of the numbers after index 3 is: 4 = 4

## left + nums[i] + left = total  -->  nums[i] = total - 2*left

def middleIndex(nums):
    left = 0
    total = sum(nums)

    for i in range(len(nums)):
        if nums[i] == total - 2*left:
            return i
        left += nums[i]
    return -1

##Time O(n) space O(1)
# nums = [1]
# print(middleIndex(nums))

##354. Russian Doll Envelopes
# Input: envelopes = [[5,4],[6,4],[6,7],[2,3]]
# # Output: 3
# # Explanation: The maximum number of envelopes you can Russian doll is 3 ([2,3] => [5,4] => [6,7]).

def russianDoll(envelops):
    ##Sort the env width in ascending and hight in descending order
    envelops.sort(key=lambda key:(key[0], -key[1]))
    table = [envelops[0][1]]
    for i in range(1, len(envelops)):
        if envelops[i][1] > table[-1]:
            table.append(envelops[i][1])
        else:
            idx = binary(envelops[i][1], table)
            table[idx] = envelops[i][1]
    return table

def binary(num, table):
    left = 0
    right = len(table)

    while left <= right:
        mid = left + (right-left)//2
        if table[mid] == num:
            return mid
        if table[mid] > num:
            right = mid - 1
        else:
            left = mid + 1
    return left

# envelopes = [[5,4],[6,4],[6,7],[2,3]]
# print(russianDoll(envelopes))
# table = [3,4,7]
# print(binary(4, table))

##300. Longest Increasing Subsequence
# Input: nums = [10,9,2,5,3,7,101,18]
# Output: 4
# Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4.

def LIS(nums):
    table = [nums[0]]
    for i in range(len(nums)):
        if nums[i] > table[-1]:
            table.append(nums[i])
        else:
            idx = binary(nums[i], table)
            table[idx] = nums[i]
    return len(table)

# nums = [7,7,7,7,7,7,7]
# print(LIS(nums))

##1996. The Number of Weak Characters in the Game
# Input: properties = [[1,5],[10,4],[4,3]]
# Output: 1
# Explanation: The third character is weak because the second character has a strictly greater attack and defense.
def weakChar(prop):
    prop.sort(key=lambda key:(key[0], -key[1]))

    maxDiff = 0
    res = 0
    for i in range(len(prop)-1, -1, -1):
        if prop[i][1] < maxDiff:
            res += 1
        maxDiff = max(maxDiff, prop[i][1])
    return res

# prop = [[1,5],[10,4],[4,3]]
# print(weakChar(prop))

##2006. Count Number of Pairs With Absolute Difference K
# Input: nums = [1,2,2,1], k = 1
# Output: 4

def absDiffK(nums, k):
    numDict = {}
    count = 0
    for i in range(len(nums)):
        x1 = nums[i] + k
        x2 = nums[i] - k
        if x1 in numDict:
            count += numDict[x1]
        if x2 in numDict:
            count += numDict[x2]
        if nums[i] in numDict:
            numDict[nums[i]] += 1
        else:
            numDict[nums[i]] = 1
    return count
##Time O(n) and space O(1)
# nums = [1,2,2,1]
# k = 1
# print(absDiffK(nums, k))


##15. 3Sum
# Input: nums = [-1,0,1,2,-1,-4]
# Output: [[-1,-1,2],[-1,0,1]]

def threeSum(nums):
    res = []
    nums.sort()

    for i in range(len(nums)):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        left = i + 1
        right = len(nums) - 1

        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total == 0:
                res.append((nums[i],nums[left], nums[right]))
                while left < right and nums[left] == nums[left+1]:
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


##49. Group Anagrams
# Input: strs = ["eat","tea","tan","ate","nat","bat"]
# Output: [["bat"],["nat","tan"],["ate","eat","tea"]]

def groupAnagram(strs):
    wordDict = {}
    for word in strs:
        tempWordList = [0]*26
        for char in word:
            pos = ord(char) - ord("a")
            tempWordList[pos] += 1
        if tuple(tempWordList) in wordDict:
            wordDict[tuple(tempWordList)].append(word)
        else:
            wordDict[tuple(tempWordList)] = [word]
    res = []
    for key in wordDict:
        res.append(wordDict[key])
    return res

##Time O(MK) M= len of strs and k is average length of word, space also O(MK)
# strs = ["a"]
# print(groupAnagram(strs))

##234. Palindrome Linked List
# Input: head = [1,2,2,1]
# Output: true

def isPelindrome(head):
    if head is None or head.next is None:
        return True
    slow = head
    fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next
    if fast:
        slow = slow.next

    revHead = revLinkedList(slow)
    while revHead:
        if revHead.bal != head.val:
            return True
        revHead = revHead.next
        head = head.next
    return True

def revLinkedList(head):
    prev = None
    while head:
        temp = head.next
        head.next = prev
        prev = head
        head = temp
    return prev

##33. Search in Rotated Sorted Array
# Input: nums = [4,5,6,7,0,1,2], target = 0
# Output: 4

def search2(nums, target):
    left = 0
    right = len(nums) - 1

    while left <= right:
        mid = left + (right-left)//2
        if nums[mid] == target:
            return mid
        if nums[mid] >= nums[left]:
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

# nums = [1]
# target = 0
# print(search2(nums, target))


##28. Implement strStr()
# Input: haystack = "hello", needle = "ll"
# Output: 2

def strStr(haystack, needle):
    if len(needle) == 0:
        return 0
    m = len(haystack)
    n = len(needle)

    if n > m:
        return -1
    else:
        for i in range(m-n+1):
            if haystack[i:i+n] == needle:
                return i
        return -1

# haystack = "aaaaa"
# needle = "bba"
# print(strStr(haystack, needle))

##85. Maximal Rectangle
# Input: matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
# Output: 6
# Explanation: The maximal rectangle is shown in the above picture.

def rectArea(heights):
    n = len(heights)
    left = [0]*n
    right = [0]*n
    stack = []
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


def maximalRectangle(matrix):
    m = len(matrix)
    n = len(matrix[0])

    for i in range(m):
        for j in range(n):
            matrix[i][j] = int(matrix[i][j])

    for i in range(1, m):
        for j in range(n):
            if matrix[i][j] != 0:
                matrix[i][j] += matrix[i-1][j]
    maxArea = 0
    for row in matrix:
        area = rectArea(row)
        maxArea = max(maxArea, area)
    return maxArea

# matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
# print(maximalRectangle(matrix))

##221. Maximal Square
# Input: matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
# Output: 4

def maximalSquare(matrix):
    table = [[0 for col in range(len(matrix[0]))] for row in range(len(matrix))]
    height = 0
    for i in range(1, len(table)):
        for j in range(1, len(table[0])):
            if matrix[i-1][j-1] == "1":
                table[i][j] = 1 + min(table[i-1][j], table[i][j-1], table[i-1][j-1])
                height = max(height, table[i][j])
    return height*height

##Time O(mn) and space O(mn)
# matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
# print(maximalSquare(matrix))


##204. Count Primes
def countPrimes(n):
    import math
    if n <= 2:
        return 0
    table = [False]*n
    limit = int(math.sqrt(n))
    for i in range(2, limit+2):
        if table[i] == False:
            for j in range(i*i, n, i):
                table[j] = True

    count = 0
    for i in range(2, len(table)):
        if table[i] == False:
            count += 1
    return count
# Time O(sqrt(n)*loglogn) space = O(n)
# print(countPrimes(10))

def comSum(nums, target):
    cSum = 0
    left = 0
    for right in range(len(nums)):
        cSum += nums[right]
        while left <= right and cSum > target:
            cSum -= nums[left]
            left += 1
        if cSum == target:
            return True
    return False

nums = [1, 4, 1, 3, 23]
target = 7
print(comSum(nums, target))