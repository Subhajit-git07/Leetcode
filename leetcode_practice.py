##7. Reverse Integer 123
def reverse(y):
    result = 0
    if y < 0:
        x = -1*y
    else:
        x = y
    while x > 0:
        result = result*10 + x%10
        x = x // 10
    if y < 0:
        return -1*result
    else:
        return result

#print(reverse(-123))

##763. Partition Labels
def partitionLabels(S):
    dict1 = {}
    for i in range(len(S)):
        dict1[S[i]] = i
    start = 0
    end = 0
    output = []
    for i in range(len(S)):
        end = max(end, dict1[S[i]])
        if i == end:
            output.append(i-start+1)
            start = i +1
    return output

#print(partitionLabels("ababcbacadefegdehijhklij"))

#56. Merge Intervals
intervals = [[1,18],[2,6],[8,10],[30,39]]
def merge(intervals):
    intervals.sort()
    i = 1
    while i < len(intervals):
        if intervals[i][0] <= intervals[i-1][1]:
            intervals[i-1][0] = min(intervals[i-1][0], intervals[i][0])
            intervals[i-1][1] = max(intervals[i-1][1], intervals[i][1])
            intervals.pop(i)
        else:
            i += 1
    return intervals

#print(merge(intervals))

##701. Insert into a Binary Search Tree

class TreeNode:
    def __init__(self, val=0):
        self.val = val
        self.left = None
        self.right = None

def preOrder(root):
    if root is None:
        return None
    print(root.val)
    if root.left:
        preOrder(root.left)
    if root.right:
        preOrder(root.right)


#             8
#            / \
#           4   10
#          / \   \
#         2   5   12
#

root = TreeNode(8)
root.left = TreeNode(4)
root.left.left = TreeNode(2)
root.left.right = TreeNode(5)
root.right = TreeNode(10)
root.right.right = TreeNode(12)

#print(preOrder(root))

def insert_BST(root, data):
    if root is None:
        root = TreeNode(data)
    if root.val > data:
        if root.left:
            insert_BST(root.left, data)
        else:
            root.left = TreeNode(data)
    if root.val < data:
        if root.right:
            insert_BST(root.right, data)
        else:
            root.right = TreeNode(data)

#insert_BST(root, 3)
#print(preOrder(root))

def search(root, data):
    if root is None:
        return False
    if root.val == data:
        return True
    if root.val > data:
        return search(root.left, data)
    if root.val < data:
        return search(root.right, data)
    return False

#print(search(root, 4))

#1161. Maximum Level Sum of a Binary Tree

def maxLevelSum(root):
    output = [root.val]
    stack = [root]
    while len(stack) != 0:
        level = []
        res = 0
        for cur in stack:
            if cur.left:
                res += cur.left.val
                level.append(cur.left)
            if cur.right:
                res += cur.right.val
                level.append(cur.right)
        if len(level) != 0:
            output.append(res)
        stack = level
    max_level = 1
    max_val = root.val
    for i in range(1,len(output)):
        if output[i] > max_val:
            max_val = output[i]
            max_level = i + 1
    return max_level

#print(maxLevelSum(root))

#Best time to Buy and sell stocks 1

##input = [7,1,3,6,3,9,0]

def maxProf(nums):
    result = 0
    min_value = nums[0]
    for i in range(1,len(nums)):
        if nums[i] < min_value:
            min_value = nums[i]
        else:
            result = max(result, nums[i] - min_value)
    return result
nums = [7,6,4,3,1]
#print(maxProf(nums))

##Best time to buy and sell stocks 2
##you can buy maximum number of stocks

def maxProf2(nums):
    result = 0
    for i in range(1, len(nums)):
        if nums[i] > nums[i-1]:
            result += nums[i] - nums[i-1]
    return result

nums = [7,6,4,3,10]
#print(maxProf2(nums))

#53. Maximum Subarray

def maxSubAarray(nums):
    max_by_end = nums[0]
    max_so_far = nums[0]

    for i in range(1, len(nums)):
        max_by_end = max(nums[i], max_by_end+nums[i])
        max_so_far = max(max_so_far, max_by_end)

    return max_so_far

nums = [5,4,-1,7,8]
#print(maxSubAarray(nums))


##763. Partition Labels
#Input: S = "ababcbacadefegdehijhklij"
#Output: [9,7,8]


def partitionLevels(S):
    dict1 = {}
    for i in range(len(S)):
        dict1[S[i]] = i

    output = []
    start = 0
    end = 0
    for i in range(len(S)):
        end = max(end, dict1[S[i]])
        if i == end:
            part = i - start + 1
            output.append(part)
            start = i + 1
    return output

S = "ababcbacadefegdehijhklij"
##print(partitionLevels(S))

#70. Climbing Stairs
def climbStair(n, memo):
    if n == 0:
        return 1
    if n in memo:
        return memo[n]
    if n < 0:
        return 0
    memo[n] = climbStair(n-1, memo) + climbStair(n-2, memo)
    return memo[n]

#print(climbStair(45, {}))


#70. Climbing Stairs --- dynamic programming

def climbStair2(n):
    if n == 1:
        return 1
    table = [0]*(n+1)
    table[1] = 1
    table[2] = 2
    for i in range(3, n+1):
        table[i] = table[i-1] + table[i-2]
    return table[n]

#print(climbStair2(45))

#62. Unique Paths

def uniquePath(m,n, memo):
    key = str(m) + "," + str(n)
    if m == 0 and n == 0:
        return 1
    if key in memo:
        return memo[key]
    if m == 1 or n == 1:
        return 1
    memo[key]= uniquePath(m-1, n, memo) + uniquePath(m, n-1, memo)
    return memo[key]

#print(uniquePath(70,30, {}))

#Bottom up approach
def uniquePath2(m,n):
    table = [[0 for col in range(n)] for row in range(m)]
    for row in range(m):
        table[row][0] = 1
    for col in range(n):
        table[0][col] = 1
    for row in range(1,m):
        for col in range(1, n):
            table[row][col] = table[row-1][col] + table[row][col-1]
    return table[m-1][n-1]

#print(uniquePath2(3,7))

##63. Unique Paths II
def uniquePathWithObstacle(matrix):
    rows = len(matrix)
    cols = len(matrix[0])

    if matrix[0][0] == 1:
        return 0
    table = [[0 for col in range(cols)] for row in range(rows)]
    table[0][0] = 1
    for row in range(1, rows):
        if row < rows and matrix[row][0] != 1:
            table[row][0] = table[row-1][0]
        else:
            table[row][0] = 0
    for col in range(1, cols):
        if col < cols and matrix[0][col] != 1:
            table[0][col] = table[0][col-1]
        else:
            table[0][col] = 0
    #print(table)
    for row in range(1, rows):
        for col in range(1, cols):
            #print(matrix[row][col])
            if row < rows and col < cols and matrix[row][col] != 1:
                table[row][col] = table[row-1][col] + table[row][col-1]
            else:
                table[row][col] = 0
    return table[rows-1][cols-1]


matrix = [[0,1],[0,0]]
#print(uniquePathWithObstacle(matrix))

##64. Minimum Path Sum
grid = [[1,2,3],[4,5,6]]
def minPathSum(grid):
    rows = len(grid)
    cols = len(grid[0])

    for row in range(1, rows):
        grid[row][0] += grid[row-1][0]
    for col in range(1, cols):
        grid[0][col] += grid[0][col-1]

    for row in range(1, rows):
        for col in range(1, cols):
            grid[row][col] += min(grid[row-1][col], grid[row][col-1])

    return grid[rows-1][cols-1]

#print(minPathSum(grid))

def dungeon(grid):
    rows = len(grid)
    cols = len(grid[0])

    if grid[rows-1][cols-1] > 0:
        grid[rows-1][cols-1] = 1
    else:
        grid[rows-1][cols-1] = 1 - grid[rows-1][cols-1]

    for row in range(rows-2, -1, -1):
        grid[row][cols-1] = max(grid[row+1][cols-1] - grid[row][cols-1], 1)

    for col in range(cols-2, -1, -1):
        grid[rows-1][col] = max(grid[rows-1][col+1] - grid[rows-1][col], 1)

    for row in range(rows-2, -1, -1):
        for col in range(cols-2, -1, -1):
            grid[row][col] = max(min(grid[row+1][col], grid[row][col+1]) - grid[row][col], 1)

    return grid[0][0]

grid = [[0]]
#print(dungeon(grid))
def oracle(s):
    result = ""
    counter = 0
    prevStr = s[0]
    for i in range(len(s)):
        curStr = s[i]
        if prevStr == curStr:
            counter += 1
            prevStr = curStr
        else:
            result += prevStr + str(counter)
            counter = 1
            prevStr = curStr
    if prevStr == curStr:
        result += prevStr + str(counter)
    else:
        result += curStr + str(counter)
    return result
##print(oracle("ABBCHBA"))

##131. Palindrome Partitioning

def partition(s):
    res = []
    part = []
    helper(s, res, part, 0)
    return res


def helper(s, res, part, i):
    if i >= len(s):
        res.append(part.copy())
        return

    for j in range(i, len(s)):
        if is_pelin(s, i, j):
            part.append(s[i:j+1])
            helper(s, res, part, j+1)
            part.pop()

def is_pelin(s, left, right):
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True

#print(partition("aab"))

def perfectSquare(n):
    if n == 1:
        return True
    mid = n // 2
    for i in range(1, mid+1):
        if i*i == n:
            return True
    return False

list1 = []
for num in range(1, 14):
    if perfectSquare(num):
        list1.append(num)
res = []
part = []
def numSquares(n , curSum, res, part, memo):
    if curSum == 0:
        res.append(part)
        return
    if curSum in memo:
        return memo[curSum]
    if curSum < 0:
        return 0

    for num in list1:
        rem  = curSum - num
        memo[rem] = numSquares(n, rem, res, part+[num], memo)

    return res

result =  numSquares(13, 13, res, part, {})

length = 13
for array in result:
    if len(array) < length:
        length = len(array)
        shortest = array

#print(shortest)

##131  Pelindrom partitioning

def pelindromPartition(s):
    result = []
    part = []
    helper(s, 0, result, part)
    return result


def helper(s, i, result, part):
    if i >= len(s):
        result.append(part.copy())
        return
    for j in range(i, len(s)):
        if is_pelin(s, i, j):
            part.append(s[i:j+1])
            helper(s, j+1, result, part)
            part.pop()

def is_pelin(s, left, right):
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True

##print(pelindromPartition("aab"))

##1480. Running Sum of 1d Array

#Input: nums = [1,2,3,4]
#Output: [1,3,6,10]

def runningSum(nums):
    output = []
    curSum = 0
    for i in range(len(nums)):
        curSum += nums[i]
        output.append(curSum)
    return output

nums = [3,1,2,10,1]
#print(runningSum(nums))

#1816. Truncate Sentence

def truncateSentance(s, k):
    s = s.split()
    str1 =  s[:k]
    return " ".join(str1)

s = "Hello how are you Contestant"
k = 4
#print(truncateSentance(s, k))

## 617. Merge Two Binary Trees

tree1 = TreeNode(1)
tree1.left = TreeNode(3)
tree1.right = TreeNode(2)
tree1.left.left = TreeNode(5)


tree2 = TreeNode(2)
tree2.left = TreeNode(1)
tree2.right = TreeNode(3)
tree2.left.right = TreeNode(4)
tree2.right.right = TreeNode(7)

def mergeTrees(tree1, tree2):
    if tree1 and tree2 is None:
        return tree1
    if tree2 and tree1 is None:
        return tree2
    if tree1 is None and tree2 is None:
        return None

    tree1.val = tree1.val + tree2.val
    tree1.left = mergeTrees(tree1.left, tree2.left)
    tree1.right = mergeTrees(tree1.right, tree2.right)

    return tree1

tree1 = mergeTrees(tree1, tree2)
#print(preOrder(tree1))

##1. Two Sum

def twoSum(nums, target):
    num_dict = {}
    for i in range(len(nums)):
        if target - nums[i] in num_dict:
            return [num_dict[target-nums[i]], i]
        else:
            num_dict[nums[i]] = i

nums = [3,3]
target = 6
##print(twoSum(nums, target))

##15. 3Sum
#Input: nums = [-1,0,1,2,-1,-4]
#Output: [[-1,-1,2],[-1,0,1]]

def threeSum(nums):
    nums.sort()
    output = []
    for i in range(len(nums)-2):
        if i>0 and nums[i] == nums[i-1]:
            continue
        left = i + 1
        right = len(nums) - 1

        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total == 0:
                output.append([nums[i], nums[left], nums[right]])
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
    return output

#nums = [0,1,5,-5,5,-6]
#print(threeSum(nums))

def sorting(nums):
    for i in range(len(nums)-1):
        for j in range(i+1,len(nums)):
            if nums[i] > nums[j]:
                nums[i], nums[j] = nums[j], nums[i]
    return nums

#nums = [3,1,4,5,-2,0]
#print(sorting(nums))

##560. Subarray Sum Equals K
#Input: nums = [1,1,1], k = 2
#Output: 2

def sumArraySum(nums, k):
    count = 0
    for i in range(len(nums)):
        res = 0
        for j in range(i, len(nums)):
            res += nums[j]
            if res == k:
                count += 1
    return count

#nums = [1,2,3]
#k = 3
#print(sumArraySum(nums, k))

##560. Subarray Sum Equals K
#Input: nums = [1,1,1], k = 2
#Output: 2
def subArraySum2(nums, k):
    dict1 = {}
    count = 0
    dict1[0] = 1
    cSum = 0
    for i in range(len(nums)):
        cSum += nums[i]
        if cSum-k in dict1:
            count += dict1[cSum-k]
        if cSum in dict1:
            dict1[cSum] += 1
        else:
            dict1[cSum] = 1
    return count
#nums = [1,2,3]
#k = 3
#print(subArraySum2(nums, k))

##523. Continuous Subarray Sum

def checkSubarraySum(nums, k):
    dict1 = {}
    dict1[0] = -1
    cSum = 0
    for i in range(len(nums)):
        cSum += nums[i]
        if k != 0:
            cSum = cSum%k
        if cSum in dict1:
            if i - dict1[cSum] >= 2:
                return True
        else:
            dict1[cSum] = 1
    return False

#nums = [23,2,6,4,7]
#k = 13
#print(checkSubarraySum(nums, k))

##713. Subarray Product Less Than K
def numSubArrayProdLessThenK(nums, k):
    count = 0
    left = 0
    prod = 1

    for i in range(len(nums)):
        prod *= nums[i]
        while prod >= k:
            prod = prod / nums[left]
            left += 1
        count += i - left + 1
    return count

#nums = [10, 5, 2, 6]
#k = 100
#print(numSubArrayProdLessThenK(nums, k))

##724. Find Pivot Index

def pivotIndex(nums):
    totalSum = sum(nums)
    leftSum = 0
    for i in range(len(nums)):
        rightSum = totalSum - leftSum - nums[i]
        if rightSum == leftSum:
            return i
        leftSum += nums[i]
    return -1
#nums = [2,1,-1]
#print(pivotIndex(nums))

#5. Longest Palindromic Substring
#s = "babad"

def longestPelin(s):
    res = ""
    for i in range(len(s)):
        odd = pelin(i, i, s)
        if len(odd) > len(res):
            res = odd
        even = pelin(i, i+1, s)
        if len(even) > len(res):
            res = even
    return res
def pelin(left, right, s):
    while left >= 0 and right < len(s) and s[left] == s[right]:
        left -= 1
        right += 1
    return s[left+1:right]

#print(longestPelin("abbd"))

##shortest-palindrome

s = "aacecaaa"
n = len(s)
rev_s = s[::-1]
#print(rev_s)
new_s = s + rev_s
#print(new_s)
dp = [0]*len(new_s)
#print(dp)
j = 0
for i in range(1, len(new_s)):
    # = dp[i-1]
    while j>0 and new_s[j] != new_s[i]:
        j = dp[j-1]
    if new_s[i] == new_s[j]:
        dp[i] = j + 1
        j += 1
res = rev_s[:n-dp[-1]] + s
#print(dp)
#print(res)

##647. Palindromic Substrings

def countSubstr(s):
    total = 0
    for i in range(len(s)):
        total += pelin2(s, i, i)
        total += pelin2(s, i, i+1)

    return total


def pelin2(s, left, right):
    count = 0
    while left >= 0 and right < len(s) and s[left] == s[right]:
        count += 1
        left -= 1
        right += 1
    return count

##print(countSubstr("aaa"))

##7. Reverse Integer

def reverseInt(num):
    res = 0
    while num > 0:
        res = res*10 + num%10
        num = num // 10
    return res
##print(reverseInt(123))

#Combination sum

#candidates = [2,3,6,7], target = 7
def combinationSum(nums, target):
    result = []
    arr = []
    nums.sort()
    helperC(0, nums, target, arr, result)
    return result

def helperC(start, nums, target, arr, result):
    if target == 0:
        result.append(arr)
        return
    if target < 0:
        return
    for i in range(start, len(nums)):
        if i > start and nums[i] == nums[i-1]:
            continue
        helperC(i + 1, nums, target - nums[i], arr+[nums[i]], result)


#nums = [10,1,2,7,6,1,5]
#nums = [2,3,6,7]
#target = 8
#print(combinationSum(nums, target))

class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def add(self, val):
        if self.head is None:
            self.head = ListNode(val)
            return
        cur = self.head
        while cur.next != None:
            cur = cur.next
        cur.next = ListNode(val)
    def print_list(self):
        if self.head is None:
            return "Hay list is empty"
        cur = self.head
        while cur != None:
            print(cur.val)
            cur = cur.next

    def remove_n_from_end(self, n):
        length = 0
        if self.head is None:
            return 0
        cur = self.head
        while cur != None:
            length += 1
            cur = cur.next
        remove_position = length - n

        cur = self.head
        prev = None
        pos = 0
        while cur != None and pos != remove_position:
            pos += 1
            prev = cur
            cur = cur.next
        prev.next = cur.next
        cur.next = None

list1 = LinkedList()
for i in range(1,6):
    list1.add(i)
#list1.remove_n_from_end(3)
#print(list1.print_list())

##32. Longest Valid Parentheses

def longestValidParen(s):
    left_bracket = 0
    right_bracket = 0
    max_length = 0

    for i in range(len(s)):
        if s[i] == '(':
            left_bracket += 1
        else:
            right_bracket += 1

        if left_bracket == right_bracket:
            max_length = max(max_length, left_bracket+right_bracket)
        elif right_bracket > left_bracket:
            left_bracket = 0
            right_bracket = 0

    left_bracket = 0
    right_bracket = 0
    for i in range(len(s)-1, -1, -1):
        if s[i] == '(':
            left_bracket += 1
        else:
            right_bracket += 1

        if left_bracket == right_bracket:
            max_length = max(max_length, left_bracket+right_bracket)
        elif left_bracket > right_bracket:
            left_bracket = 0
            right_bracket = 0

    return max_length

#print(longestValidParen(')()())'))

##34. Find First and Last Position of Element in Sorted Array

def searchRange(nums, target):
    left = searchHelper(nums, target, True)
    right = searchHelper(nums, target, False)
    return [left, right]


def searchHelper(nums, target, leftbias):
    left = 0
    right = len(nums)

    i = -1

    while left <= right:
        mid = left + (right-left)// 2
        if nums[mid] > target:
            right = mid - 1
        elif nums[mid] < target:
            left = mid + 1
        else:
            i = mid
            if leftbias:
                right = mid - 1
            else:
                left = mid + 1
    return i

#nums = [5,7,7,8,9,10]
#target = 9
#print(searchRange(nums, target))

#35. Search Insert Position

def searchInsert(nums, target):
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

#nums = [1, 3, 5, 6]
#target = 4
#print(searchInsert(nums, target))

#41. First Missing Positive

def firstMissingPositive(nums):
    if len(nums) == 0:
        return 1
    nums_dict = {}
    for num in nums:
        nums_dict[num] = True
    for i in range(1, len(nums)+1):
        if i not in nums_dict:
            return i
    return i + 1

#nums = [7,8,9,11,12]
#print(firstMissingPositive(nums))

def firstMissingPositive1(nums):
    n = len(nums)
    for i in range(len(nums)):
        if nums[i] <= 0 or nums[i] > n:
            nums[i] = n + 1

    for i in range(len(nums)):
        cur = abs(nums[i])
        if cur > n:
            continue
        cur = cur - 1
        if nums[cur] > 0:
            nums[cur] = -nums[cur]

    for i in range(len(nums)):
        if nums[i] > 0:
            return i + 1

    return n + 1

#nums = [7,8,9,1,11,12]
#print(firstMissingPositive1(nums))




