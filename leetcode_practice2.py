##1. Two Sum
def twoSum(nums, target):
    nums_dict = {}
    for i in range(len(nums)):
        if target - nums[i] in nums_dict:
            return [nums_dict[target-nums[i]], i]
        else:
            nums_dict[nums[i]] = i

#nums = [3,2,4,6]
#target = 6
#print(twoSum(nums, target))

##2. Add Two Numbers
class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None

def addTwoNumbers(l1, l2):
    dummy = ListNode(0)
    temp = dummy
    carry = 0

    while l1 or l2 or carry:
        if l1:
            v1 = l1.val
        else:
            v1 = 0

        if l2:
            v2 = l2.val
        else:
            v2 = 0
        value = v1 + v2 + carry
        carry = value // 10
        value = value % 10
        temp.next = ListNode(value)
        temp = temp.next
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None


    return dummy.next

##3. Longest Substring Without Repeating Characters
#Input: s = "abcabcbb"
#Output: 3
def lengthOfLongestSubstr(s):
    seen = {}
    start = 0
    max_length = 0

    for i in range(len(s)):
        if s[i] in seen:
            start = max(start, seen[s[i]]+1)
        seen[s[i]] = i
        max_length = max(max_length, i - start + 1)

    return max_length
#s = "pwwkew"
#print(lengthOfLongestSubstr(s))

##5. Longest Palindromic Substring
#Input: s = "babad"
#Output: "bab"
def longestPelindrom(s):
    result = ""
    for i in range(len(s)):
        odd = pelin(i, i, s)
        if len(odd) > len(result):
            result = odd
        even = pelin(i, i+1, s)
        if len(even) > len(result):
            result = even
    return result


def pelin(left, right, s):
    while left >= 0 and right < len(s) and s[left] == s[right]:
        left -= 1
        right += 1
    return s[left+1:right]

##Time complexity is O(n^2) and space complexity is O(1)
#s = "ac"
#print(longestPelindrom(s))

##11. Container With Most Water
def maxArea(height):
    left = 0
    right = len(height) - 1
    max_area = 0
    while left < right:
        area = min(height[left], height[right]) * (right-left)
        max_area = max(max_area, area)
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return max_area

#height = [1,2,1]
#print(maxArea(height))

##14. Longest Common Prefix
#Input: strs = ["flower","flow","flight"]
#Output: "fl"

##Fine the lowest length og string
def lowestLength(strs):
    low_length = len(strs[0])
    for i in range(1, len(strs)):
        low_length = min(low_length, len(strs[i]))
    return low_length
strs = ["dog","racecar","car"]
#print(lowestLength(strs))

def longestCommonPrefix(strs):
    length = lowestLength(strs)
    result = ""
    s = strs[0]
    for i in range(length):
        for j in range(1, len(strs)):
            if strs[j][i] != s[i]:
                return result
        result += s[i]
    return result
#n = min length of string of strings, m = length og strs
#Time complexity is O(m.n) and space complexity is O(1)
#print(longestCommonPrefix(strs))

##15. 3Sum
#Input: nums = [-1,0,1,2,-1,-4]
#Output: [[-1,-1,2],[-1,0,1]]

def threeSum(nums):
    nums.sort()
    output = []
    for i in range(len(nums) - 2):
        left = i + 1
        right = len(nums) - 1
        if i > 0 and nums[i] == nums[i-1]:
            continue
        while left < right and nums[left] == nums[left + 1]:
            left += 1
        while left < right and nums[right] == nums[right - 1]:
            right -= 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total == 0:
                output.append([nums[i], nums[left], nums[right]])

                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1

    return output

#nums = [0,0,0]      #-4 -1 -1 0 1 2
#print(threeSum(nums))

##16. 3Sum Closest
def threeSumClosest(nums, target):
    diff = float('inf')
    nums.sort()
    for i in range(len(nums) - 2):
        left = i + 1
        right = len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if abs(target-total) < abs(diff):
                diff = target - total
            if total < target:
                left += 1
            else:
                right -= 1
    return target - diff

#nums = [-1,2,1,-4]
#target = 1
#print(threeSumClosest(nums, target))
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
                bottom = stack.pop()
                if not is_valid(bottom, s[i]):
                    return False
    if len(stack) == 0:
        return True
    else:
        return False

def is_valid(p1, p2):
    if p1 == "(" and p2 == ")":
        return True
    elif p1 == "{" and p2 == "}":
        return True
    elif p1 == "[" and p2 == "]":
        return True
    else:
        return False

#s = "{[]}"
#print(validParen(s))

##21. Merge Two Sorted Lists
def merge(l1, l2):
    dummy = ListNode(0)
    temp = dummy
    while l1 and l2:
        if l1.val < l2.val:
            temp.next = l1
            l1 = l1.next
        else:
            temp.next = l2
            l2 = l2.next
        temp = temp.next

    while l1:
        temp.next = l1
        l1 = l1.next
        temp = temp.next

    while l2:
        temp.next = l2
        l2 = l2.next
        temp = temp.next

    return dummy.next

##23. Merge k Sorted Lists

def mergeKSortedList(lists):
    while len(lists) > 1:
        merged = []
        for i in range(0, len(lists), 2):
            l1 = lists[i]
            if i+1 < len(lists):
                l2 = lists[i+1]
            else:
                l2 = None
            merged.append(merge(l1, l2))
        lists = merged

    return lists[0]

##24. Swap Nodes in Pairs
def swapPairs(head):
    d1 = ListNode(0)
    d = d1
    d.next = head
    while d.next and d.next.next:
        p = d.next
        q = d.next.next

        d.next = q
        p = d.next.next
        q.next = p

        d = p
    return d1.next

##26. Remove Duplicates from Sorted Array
def removeDuplicates(nums):
    i = 0
    for j in range(1, len(nums)):
        if nums[j] != nums[i]:
            i += 1
            nums[i] = nums[j]
    return i + 1

#nums = [0,0,1,1,1,1,2,3,3,3,4]
#print(removeDuplicates(nums))

##27. Remove Element
def removeElement(nums, val):
    i = 0
    for j in range(len(nums)):
        if nums[j] != val:
            nums[i] = nums[j]
            i += 1
    return i

#nums = [0,1,2,2,3,0,4,2]
#val = 2
#print(removeElement(nums, val))

##32. Longest Valid Parentheses
def longestValidParen(s):
    left = 0
    right = 0
    max_length = 0
    cur = ""
    res = ""
    for i in range(len(s)):
        if s[i] == "(":
            left += 1
        else:
            right += 1
        cur += s[i]

        if left == right:
            #max_length = max(max_length, left+right)
            if len(cur) > max_length:
                max_length = len(cur)
                res = cur
        elif right > left:
            left = 0
            right = 0
            cur = ""

    left = 0
    right = 0
    cur = ""
    for i in range(len(s)-1, -1, -1):
        if s[i] == "(":
            left += 1
        else:
            right += 1
        cur = s[i] + cur


        if left == right:
            #max_length = max(max_length, left + right)
            if len(cur) > max_length:
                max_length = len(cur)
                res = cur
        elif left > right:
            left = 0
            right = 0
            cur = ""

    return res
##s = "(()()()"
##print(longestValidParen(s))

##33. Search in Rotated Sorted Array
def search1(nums, target):
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
#nums = [1]
#target = 0
#print(search1(nums, target))

##35. Search Insert Position
def searchInsert(nums, target):
    left = 0
    right = len(nums) - 1

    while left <= right:
        mid = left + (right-left)//2
        if nums[mid] == target:
            return mid
        elif nums[mid] > target:
            right = mid -1
        else:
            left = mid + 1
    return left

#nums = [1]
#target = 0
#print(searchInsert(nums, target))

##39. Combination Sum
#Input: candidates = [2,3,6,7], target = 7
#Output: [[2,2,3],[7]]

def combinationSum(nums, target):
    result = []
    arr = []
    helper(nums, target, result, arr)
    return result

def helper(nums, target, result, arr):
    if target == 0:
        result.append(arr)
        return
    if target < 0:
        return

    for i in range(len(nums)):
        helper(nums[i:], target-nums[i], result, arr+[nums[i]])

#nums = [2,3,6,7]
#target = 7
#print(combinationSum(nums, target))

##40. Combination Sum II
def combinationSum2(nums, target):
    result = []
    arr = []
    nums.sort()
    helper2(0, nums, target, result, arr)
    return result

def helper2(start, nums, target, result, arr):
    if target == 0:
        result.append(arr)
        return
    if target < 0:
        return

    for i in range(start, len(nums)):
        if i>start and nums[i] == nums[i-1]:
            continue
        helper2(i+1, nums, target-nums[i], result, arr+[nums[i]])
#nums = [2,5,2,1,2]
#target = 5
#print(combinationSum2(nums, target))

##41. First Missing Positive
##Using space
def firstMissingPositive(nums):
        n = len(nums)
        nums_dict = {}
        for num in nums:
            nums_dict[num] = True

        for i in range(1, n+1):
            if i not in nums_dict:
                return i
        return n + 1

##Without using space
#[3,4,-1,1]
#[3,4,5,1]
#[-3,  4  -5, -1]
##[1, 1]
def firstMissingPositive2(nums):
    n = len(nums)
    for i in range(len(nums)):
        if nums[i] <= 0 or nums[i] > n:
            nums[i] = n + 1
    print(nums)

    for i in range(len(nums)):
        cur = abs(nums[i])
        if cur > n:
            continue
        cur = cur - 1
        if nums[cur] > 0:
            nums[cur] = -nums[cur]
    print(nums)

    for i in range(len(nums)):
        if nums[i] > 0:
            return i + 1
    return n + 1
#nums = [3,4,-1,1]
#print(firstMissingPositive2(nums))

##42. Trapping Rain Water
def trap(nums):
    result = 0
    left_max = 0
    right_max = 0
    left = 0
    right = len(nums) - 1

    while left < right:
        left_max = max(left_max, nums[left])
        right_max = max(right_max, nums[right])

        if left_max <= right_max:
            result += left_max - nums[left]
            left += 1
        else:
            result += right_max - nums[right]
            right -= 1

    return result

#nums = [4,2,0,3,2,5]
#print(trap(nums))

#53. Maximum Subarray
#Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
#Output: 6

def maxSubArraySum(nums):
    max_by_end = nums[0]
    max_so_far = nums[0]

    for i in range(1, len(nums)):
        max_by_end = max(nums[i], max_by_end+nums[i])
        max_so_far = max(max_so_far, max_by_end)

    return max_so_far
#nums = [5,4,-1,7,8]
#print(maxSubArraySum(nums))

##56. Merge Intervals
#Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
#Output: [[1,6],[8,10],[15,18]]

def mergeInterval(intervals):
    i = 1
    while len(intervals) > i:
        if intervals[i-1][1] >= intervals[i][0]:
            intervals[i-1][1] = min(intervals[i][1], intervals[i-1][1])
            intervals[i-1][1] = max(intervals[i-1][1], intervals[i][1])
            intervals.pop(i)
        else:
            i += 1
    return intervals

#intervals = [[1,3],[2,6],[8,10],[15,18]]
#print(mergeInterval(intervals))

##62. Unique Paths
def uniquePaths(m,n):
    grid = [[0 for col in range(n)] for row in range(m)]
    grid[0][0] = 1
    for row in range(1,m):
        grid[row][0] = 1
    for col in range(1,n):
        grid[0][col] = 1
    for row in range(1,m):
        for col in range(1,n):
            grid[row][col] = grid[row-1][col] + grid[row][col-1]
    return grid[m-1][n-1]

#print(uniquePaths(7,9))

##63. Unique Paths II
def uniquePathWithObstacle(grid):
    m = len(grid)
    n = len(grid[0])

    if grid[0][0] == 1:
        return 0
    grid[0][0] = 1
    for row in range(1, m):
        if grid[row][0] == 1:
            grid[row][0] = 0
        else:
            grid[row][0] = grid[row - 1][0]

    for col in range(1,n):
        if grid[0][col] == 1:
            grid[0][col] = 0
        else:
            grid[0][col] = grid[0][col-1]

    for row in range(1,m):
        for col in range(1,n):
            if grid[row][col] == 1:
                grid[row][col] = 0
            else:
                grid[row][col] = grid[row-1][col] + grid[row][col-1]
    return grid[m-1][n-1]

#grid = [[0,1],[0,0]]
#print(uniquePathWithObstacle(grid))

def minPathSum(grid):
    m = len(grid)
    n = len(grid[0])

    for row in range(1,m):
        grid[row][0] = grid[row][0] + grid[row-1][0]
    for col in range(1,n):
        grid[0][col] = grid[0][col] + grid[0][col-1]

    for row in range(1,m):
        for col in range(1,n):
            grid[row][col] = min((grid[row][col]+grid[row-1][col]), grid[row][col]+grid[row][col-1])
    return grid[m-1][n-1]

#grid = [[1,2,3],[4,5,6]]
#print(minPathSum(grid))

##70. Climbing Stairs
def climbStairs(n):
    table = [0]*(n+1)
    table[1] = 1
    table[2] = 2
    for i in range(3, len(table)):
        table[i] = table[i-1] + table[i-2]
    return table[n]

#n = 40
#print(climbStairs(n))

##81. Search in Rotated Sorted Array II
#Input: nums = [2,5,6,0,0,1,2], target = 0
#Output: true

def search2(nums, target):
    left = 0
    right = len(nums) - 1

    while left <= right:
        mid = left + (right-left)//2
        if nums[mid] == target:
            return True
        while mid > left and nums[mid] == nums[left]:
            left += 1

        while right > mid and nums[mid] == nums[right]:
            right -= 1

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
    return False
#nums = [2,5,6,0,0,1,2]
#target = 3
#print(search2(nums, target))

##121. Best Time to Buy and Sell Stock
def maxProfit(nums):
    min_profit = nums[0]
    result = 0
    for i in range(1, len(nums)):
        if nums[i] < min_profit:
            min_profit = nums[i]
        else:
            current_profit = nums[i] - min_profit
            result = max(result, current_profit)

    return result

##nums = [7,6,4,3,1]
##print(maxProfit(nums))

##152. Maximum Product Subarray
#Input: nums = [2,3,-2,4]
#Output: 6

def maxProduct(nums):
    prev_min_prod = nums[0]
    prev_max_prod = nums[0]
    result = nums[0]

    for i in range(1, len(nums)):
        cur_min_prod = min(prev_min_prod*nums[i], prev_max_prod*nums[i], nums[i])
        cur_max_prod = max(prev_min_prod*nums[i], prev_max_prod*nums[i], nums[i])

        result = max(result, cur_max_prod)

        prev_min_prod = cur_min_prod
        #prev_min_prod = min(prev_min_prod, cur_min_prod)
        prev_max_prod = cur_max_prod
        #prev_max_prod = max(prev_max_prod, cur_max_prod)

    return result
#nums = [2,3,-2,4]
#print(maxProduct(nums))

##153. Find Minimum in Rotated Sorted Array
def findMin(nums):
    left = 0
    right = len(nums) - 1
    if nums[right] > nums[left]:
        return nums[left    ]
    while left <= right:
        mid = left + (right-left)// 2
        if nums[mid] < nums[mid+1]:
            return nums[mid]
        if nums[mid] > nums[mid+1]:
            return nums[mid+1]
        if nums[mid-1] > nums[mid]:
            return nums[mid]
        if nums[mid] > nums[left]:
            left = mid + 1
        else:
            right = mid - 1
#nums =  [11,13,15,17]
#print(findMin(nums))
#167. Two Sum II - Input array is sorted
#Input: numbers = [2,7,11,15], target = 9
#Output: [1,2]
def twoSum2(nums, target):
    left = 0
    right = len(nums) - 1
    while left < right:
        mid = left +(right-left)//2
        if nums[left] + nums[right] == target:
            return [left+1, right+1]
        elif nums[left] + nums[right] < target:
            left += 1
        else:
            right -= 1
#nums = [2,3,4]
#target = 5
#print(twoSum2(nums, target))

##174. Dungeon Game
#-2 -3  3
#-5 -10 1
#10 30 -5

#7  5   2
#6  11  5
#1   1  6

def dungeon(grid):
    m = len(grid)
    n = len(grid[0])

    if grid[m-1][n-1] <= 0:
        grid[m-1][n-1] = 1 - grid[m-1][n-1]
    else:
        grid[m-1][n-1] = 1

    for row in range(m-2, -1, -1):
        grid[row][n-1] = max((grid[row+1][n-1] - grid[row][n-1]), 1)

    for col in range(n-2, -1 , -1):
        grid[m-1][col] = max((grid[m-1][col+1] - grid[m-1][col]), 1)

    for row in range(m-2, -1, -1):
        for col in range(n-2, -1, -1):
            grid[row][col] = max(min((grid[row+1][col]-grid[row][col]), (grid[row][col+1]-grid[row][col])), 1)

    return grid[0][0]

#grid = [[-2,-3,3],[-5,-10,1],[10,30,-5]]
#print(dungeon(grid))

##198. House Robber
#Input: nums = [2,7,9,3,1]
#Output: 12

#2 7 11 11 12

def rob(nums):
    if len(nums) == 1:
        return nums[0]
    if len(nums) == 2:
        return max(nums[0], nums[1])
    table = [0]*len(nums)
    table[0] = nums[0]
    table[1] = nums[1]
    for i in range(2, len(table)):
        table[i] = max(table[i-1], nums[i]+table[i-2])
    return table[-1]
#nums = [1,2,3,1]
#print(rob(nums))

##209. Minimum Size Subarray Sum
##Input: target = 7, nums = [2,3,1,2,4,3]
##Output: 2

def minSubArrayLen(nums, target):
    result = len(nums) + 1
    subArraySum = 0
    left = 0
    for right in range(len(nums)):
        subArraySum += nums[right]
        while left <= right and subArraySum >= target:
            result = min(result, right - left + 1)
            subArraySum -= nums[left]
            left += 1
    if result == len(nums) + 1:
        return 0
    else:
        return result
#nums = [1,1,1,1,1,1,1,1]
#target = 11
#print(minSubArrayLen(nums, target))

##216. Combination Sum III
#Input: k = 3, n = 9
#Output: [[1,2,6],[1,3,5],[2,3,4]]
def combinationSum3(k, n):
    nums = [1,2,3,4,5,6,7,8,9]
    arr = []
    result = []
    helper3(nums, k, n, arr, result)
    return result
def helper3(nums,k, n, arr, result):
    if len(arr) == k and sum(arr) == n:
        result.append(arr)
        return
    if len(arr) > k:
        return
    for i in range(len(nums)):
        if nums[i] not in arr:
            helper3(nums[i:], k, n, arr+[nums[i]], result)

#print(combinationSum3(9, 45))

##238. Product of Array Except Self
#Input: nums = [1,2,3,4]
#Output: [24,12,8,6]

def productExceptSelf(nums):
    n = len(nums)
    left = [1]*n
    right = [1]*n

    for i in range(1, len(nums)):
        left[i] = left[i-1]*nums[i-1]
    for i in range(len(nums)-2, -1, -1):
        right[i] = right[i+1]*nums[i+1]

    print(left)
    print(right)
    result = []
    for i in range(n):
        print(i)
        result.append(left[i]*right[i])

    return result
#nums = [1,2,3,4]
#print(productExceptSelf(nums))

##300. Longest Increasing Subsequence
##Input: nums = [10,9,2,5,3,7,101,18]
##Output: 4
def lengthOfLIS(nums):
    table = [1]*len(nums)
    for i in range(1, len(nums)):
        maxLen = 0
        for j in range(i):
            if nums[i] > nums[j]:
                maxLen = max(maxLen, table[j])
        table[i] = maxLen + 1
    return table[-1]
##nums = [7,7,7,7,7,7,7]
##print(lengthOfLIS(nums))

def binary_insert(nums, target):
    left = 0
    right = len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return mid
        elif target > nums[mid]:
            left = mid + 1
        else:
            right = mid - 1
    return left
## Longest increasing subsequence using binary search which is nlogn
def lengthOfLIS2(nums):
    table = [nums[0]]
    for i in range(1, len(nums)):
        if table[-1] < nums[i]:
            table.append(nums[i])
        else:
            ind = binary_insert(table, nums[i])
            table[ind] = nums[i]
    return len(table)
##nums = [0,1,0,3,2,3]
##print(lengthOfLIS2(nums))

#1749. Maximum Absolute Sum of Any Subarray
#Input: nums = [2,-5,1,-4,3,-2]
#Output: 8

def maxAbsoluteSum(nums):
    max_by_end = nums[0]
    min_by_end = nums[0]
    max_so_far = abs(nums[0])

    for i in range(1, len(nums)):
        max_by_end = max(max_by_end+nums[i], nums[i])
        min_by_end = min(min_by_end+nums[i], nums[i])
        max_so_far = max(max_so_far, abs(max_by_end), abs(min_by_end))

    return max_so_far

#nums = [2,-5,1,-4,3,-2]
#print(maxAbsoluteSum(nums))

##763. Partition Labels
#Input: s = "ababcbacadefegdehijhklij"
#Output: [9,7,8]

def partitionLevels(s):
    start = 0
    end = 0
    res = []
    str_dict = {}
    for i in range(len(s)):
        str_dict[s[i]] = i

    for i in range(len(s)):
        end = max(end, str_dict[s[i]])
        if i == end:
            res.append(i-start+1)
            start = i + 1
    return res
##s = "ababcbacadefegdehijhklij"
##print(partitionLevels(s))

##974. Subarray Sums Divisible by K
#Input: nums = [4,5,0,-2,-3,1], k = 5
#4  5 0 -2 -3 1
#{0:1,4:4,2:1,  }
#c = 1 +2 + 3 + 1

def subArrayDivByK(nums, k):
    nums_dict = {0:1}
    cSum = 0
    count = 0
    for num in nums:
        cSum += num
        rem = cSum % k
        if rem not in nums_dict:
            nums_dict[rem] = 1
        else:
            count += nums_dict[rem]
            nums_dict[rem] += 1
    return count

#nums = [4,5,0,-2,-3,1]
#k = 5
#print(subArrayDivByK(nums, k))
##724. Find Pivot Index
#Input: nums = [1,7,3,6,5,6]
#Output: 3

def pivotIndex(nums):
    total = sum(nums)
    cSum = 0
    for i in range(len(nums)):
        if cSum == total - nums[i] - cSum:
            return i
        cSum += nums[i]
    return -1

#nums = [2,1,-1]
#print(pivotIndex(nums))

##713. Subarray Product Less Than K
#Input: nums = [10, 5, 2, 6], k = 100
#Output: 8

def subArrayProdLessK(nums, k):
    left = 0
    prod = 1
    count = 0
    output = []
    for right in range(len(nums)):
        prod *= nums[right]
        while prod >= k:
            prod = prod / nums[left]
            left += 1
        count += right - left + 1

    return count
#nums = [10, 5, 2, 6]
#k = 100
#print(subArrayProdLessK(nums, k))

##647. Palindromic Substrings
def helper5(s, left, right):
    res = []
    count = 0
    while left >= 0 and right < len(s) and s[left] == s[right]:
        res.append(s[left:right+1])
        count += 1
        left -= 1
        right += 1
    #return res
    return count

def pelinSubStr(s):
    output = 0
    for i in range(len(s)):
        res1 = helper5(s,i,i)
        res2 = helper5(s,i,i+1)
        output += res1
        output += res2
    return output

#s = "aaa"
#print(pelinSubStr(s))

##560. Subarray Sum Equals K
##Input: nums = [1,2,3], k = 3
##Output: 2

def subArraySumEqualk(nums, k):
    num_dict = {0:1}
    count = 0
    cSum = 0
    for num in nums:
        cSum += num
        if cSum - k in num_dict:
            count += num_dict[cSum-k]
        if cSum in num_dict:
            num_dict[cSum] += 1
        else:
            num_dict[cSum] = 1
    return count

#nums = [1,2,3]
#k = 3
#print(subArraySumEqualk(nums, k))

##523. Continuous Subarray Sum
#Input: nums = [23,2,4,6,7], k = 6
#Output: true
def checkSubArraySum(nums, k):
    num_dict = {0:-1}
    cSum = 0
    for i in range(len(nums)):
        cSum += nums[i]
        if k != 0:
            cSum = cSum%k
        if cSum in num_dict:
            if i - num_dict[cSum] >= 2:
                return True
        else:
            num_dict[cSum] = i
    return False
#nums = [23,2,4,6,7]
#k = 6
#print(checkSubArraySum(nums, k))

##75. Sort Colors
#Input: nums = [2,0,2,1,1,0]
#Output: [0,0,1,1,2,2]

def sortColors(nums):
    left = 0
    right = len(nums) - 1
    current = 0

    while current <= right:
        if nums[current] == 0:
            nums[current], nums[left] = nums[left], nums[current]
            left += 1
            current += 1
        elif nums[current] == 2:
            nums[current], nums[right] = nums[right], nums[current]
            right -= 1
        else:
            current += 1

    return nums
#nums = [2,0,2,1,1,0]
#print(sortColors(nums))

##209. Minimum Size Subarray Sum
#Input: target = 7, nums = [2,3,1,2,4,3]
#Output: 2

def minSizeSubArray(nums, target):
    cSum = 0
    left = 0
    right = 0
    output = [0]*(len(nums)+1)
    res = len(nums) + 1
    for right in range(len(nums)):
        cSum += nums[right]
        while cSum >= target:
            #res = min(res, right-left+1)
            temp = nums[left:right+1]
            if len(output) > len(temp):
                output = temp
            cSum -= nums[left]
            left += 1
    if len(output) > len(nums):
        return 0
    else:
        return output

#nums = [1,1,1,1,1,1,1,1]
#target = 11
#print(minSizeSubArray(nums, target))

##78. Subsets
#Input: nums = [1,2,3]
#Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]

def subsets(nums):
    output = [[]]
    for num in nums:
        print(output)
        output = output + [lst+[num] for lst in output]
    return output

#nums = [1,2,3]
#print(subsets(nums))

#90. Subsets II

def subsets2(nums):
    output = [[]]
    for i in range(len(nums)):
        #if i > 0 and nums[i-1] == nums[i]:
           #continue
        for lst in output:
            cur = lst + [nums[i]]
            if cur not in output:
                output = output + [cur]
    return output
#nums = [1,2,2]
#print(subsets2(nums))

##81. Search in Rotated Sorted Array II
#Input: nums = [2,5,6,0,0,1,2], target = 0
#Output: true
def search2(nums, target):
    left = 0
    right = len(nums) - 1

    while left <= right:
        mid = left + (right-left)//2
        if nums[mid] == target:
            return True
        while mid > left and nums[mid] == nums[left]:
            left += 1
        while right > mid and nums[mid] == nums[right]:
            right -= 1
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
    return False
#nums = [2,5,6,0,0,1,2]
#target = 3
#print(search2(nums, target))

##84. Largest Rectangle in Histogram
#Input: heights = [2,1,5,6,2,3]
#Output: 10

def largestRectangleArea(nums):
    n = len(nums)
    left =  [0]*n
    right = [0]*n
    stack = []
    for i in range(len(nums)):
        if len(stack) == 0:
            stack.append(i)
            left[i] = 0
        else:
            while len(stack) != 0 and nums[i] <= nums[stack[-1]]:
                stack.pop()
            if len(stack) == 0:
                left[i] = 0
            else:
                left[i] = stack[-1] + 1
            stack.append(i)

    while len(stack) != 0:
        stack.pop()

    for i in range(len(nums)-1, -1 , -1):
        if len(stack) == n-1:
            stack.append(i)
            right[i] = n - 1
        else:
            while len(stack) != 0 and nums[i] <= nums[stack[-1]]:
                stack.pop()
            if len(stack) == 0:
                right[i] = n - 1
            else:
                right[i] = stack[-1] - 1
            stack.append(i)

    max_area = 0
    for i in range(len(nums)):
        max_area = max(max_area, (right[i]-left[i]+1)*nums[i])
    return max_area

#nums = [1,1]
#print(largestRectangleArea(nums))

#85. Maximal Rectangle
#Input: matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
#Output: 6

def maximalRectangle(matrix):
    for row in range(len(matrix)):
        for col in range(len(matrix[0])):
            matrix[row][col] = int(matrix[row][col])

    for row in range(1, len(matrix)):
        for col in range(len(matrix[0])):
            if matrix[row][col] != 0:
                matrix[row][col] += matrix[row-1][col]
    max_area = 0
    #print(matrix)
    for i in range(len(matrix)):
        max_area = max(max_area, largestRectangleArea(matrix[i]))
    return max_area

#matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
#print(maximalRectangle(matrix))

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

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

def inorder(root):
    if root is None:
        return None
    if root.left:
        inorder(root.left)
    print(root.val)
    if root.right:
        inorder(root.right)

#print(inorder(root))
#96. Unique Binary Search Trees

#Input: n = 3
#Output: 5

def numTrees(n):
    if n < 2:
        return 1
    table = [0]*(n+1)
    table[0] = 1
    table[1] = 1
    for i in range(2, len(table)):
        for j in range(i):
            table[i] += table[j]*table[i-j-1]
    return table[n]

#print(numTrees(4))

##105. Construct Binary Tree from Preorder and Inorder Traversal

def buildTree(preorder, inorder):
    if inorder:
        root = TreeNode(preorder.pop(0))
        index = inorder.index(root.val)
        root.left = bulidTree(preorder, inorder[:index])
        root.right = buildTree(preorder, inorder[index+1:])
        return root

#108. Convert Sorted Array to Binary Search Tree

def sortedArrayToBST(nums):
    if len(nums) == 0:
        return None
    mid = len(nums)//2
    root = TreeNode(nums[mid])
    root.left = sortedArrayToBST(nums[:mid])
    root.right = sortedArrayToBST((nums[mid+1:]))
    return root

##The painter’s partition problem
##Using binary search

#Input : k = 2, A = {10, 20, 30, 40} 
#Output : 60.

def numberOfPainters(nums, maxTime):
    total = 0
    num_of_painters = 1

    for num in nums:
        total += num
        if total > maxTime:
            total = num
            num_of_painters += 1
    return num_of_painters

def painterPartition(nums, k):
    left = max(nums)
    right = sum(nums)

    while left <= right:
        mid = (left+right)//2
        required_painters = numberOfPainters(nums, mid)
        if required_painters > k:
            left = mid + 1
        else:
            right = mid - 1
    return left

#nums = [10,20,30,40]
#k = 2
#print(painterPartition(nums, k))

#126. Word Ladder II
def findLadders(beginWord, endWord, wordList):
    level = 1
    res = []
    queue = [beginWord]
    wordList = set(wordList)
    alpha = 'abcdefghijklmnopqrstuvwxyz'

    while len(queue) != 0:
        next = []

        for cur in queue:
            for i in range(len(cur)):

                for ch in alpha:
                    word_list = []
                    next_word = cur[:i] + ch + cur[i + 1:]
                    if next_word in wordList:
                        wordList.remove(next_word)
                        word_list.append(next_word)
                        next.append(next_word)
                    if next_word == endWord:
                        word_list.append(next_word)
                        res.append(word_list)
        queue = next
        level += 1
    return res

#beginWord = "hit"
#endWord = "cog"
#wordList = ["hot","dot","dog","lot","log","cog"]
#print(findLadders(beginWord, endWord, wordList))

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

##112. Path Sum
def hasPathSum(root, targetSum):
    if root is None:
        return False

    if root.val == targetSum and root.left is None and root.right is None:
        return True

    return hasPathSum(root.left, targetSum-root.val) and hasPathSum(root.right, targetSum-root.val)

##199. Binary Tree Right Side View

def rightSideView(root):
    if root is None:
        return None
    result = []
    stack = [root]
    while len(stack) != 0:
        level = []
        result.append(stack[-1].val)
        for node in stack:
            if node.left:
                level.append(node.left)
            if node.right:
                level.append(node.right)
        stack = level
    return result

##Time complexity is O(n)

def search5(nums, target):
    if nums[-1] < target:
        nums.append(target)
    left = 0
    right = len(nums) - 1

    while left<=right:
        mid = left + (right-left)//2
        if nums[mid] == target:
            nums[mid] = target
            return nums
        if nums[mid] > target:
            right = mid - 1
        else:
            left = mid + 1
    nums[left] = target
    return nums
#nums = [1,3,7,8]
#target = 9

#print(search5(nums, target))
#n = 4
#board = [["." for col in range(n)] for row in range(n)]
#print(board)

def canConstruct(board):
    result = []
    for i in range(len(board)):
        result.append("".join(board[i]))
    return result
#print(canConstruct(board))

##Backtrackking

def csum(nums, target):
    result = []
    nums.sort(reverse=True)
    backtracking(nums, target, result, [], 0, len(nums))
    return result

def backtracking(nums, target, result, temp, start, end):
    if target == 0:
        result.append(temp[:])
    elif target >0:
        for i in range(start, end):
            temp.append(nums[i])
            backtracking(nums, target-nums[i], result, temp, i, end)
            temp.pop()


##nums = [1,3,4,9,2]
#target = 9
#print(csum(nums, target))

##==================================================================================================================

#CombinationSum2

def combinationSum2(nums, target):
    result = []
    temp = []
    nums.sort()
    helper_1(nums, target, result, temp, 0)
    return result

def helper_1(nums, target, result, temp, start):
    if target == 0:
        result.append(temp[:])
        return
    if target < 0:
        return

    for i in range(start, len(nums)):
        if i>start and nums[i] == nums[i-1]:
            continue
        temp.append(nums[i])
        helper_1(nums, target-nums[i], result, temp, i+1)
        temp.pop()


#nums= [10,1,2,7,6,1,5]
#target = 8
#print(combinationSum2(nums, target))





















