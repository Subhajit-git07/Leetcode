# Given a list of integers, write a function that returns the largest sum of non-adjacent numbers. Numbers can be 0
# or negative.
# For example, [2, 4, 6, 2, 5] should return 13, since we pick 2, 6, and 5. [5, 1, 1, 5] should return 10, since
# we pick 5 and 5.
# Follow-up: Can you do this in O(N) time and constant space?

def largestSumNonAdj(nums):
    nums[1] = max(nums[0], nums[1])

    for i in range(2, len(nums)):
        nums[i] = max(nums[i]+nums[i-2], nums[i-1])
    return nums[-1]

# nums = [5, 1, 1, 5]
# print(largestSumNonAdj(nums))

##Queue using two stacks

class queue:
    def __init__(self):
        self.s1 = []
        self.s2 = []

    def enque(self, val):
        while len(self.s1) != 0:
            self.s2.append(self.s1.pop())

        self.s1.append(val)

        while len(self.s2) != 0:
            self.s1.append(self.s2.pop())

    def deque(self):
        if len(self.s1) == 0:
            return "Queue is empty"
        x = self.s1.pop()
        return x

# q = queue()
# q.enque(2)
# q.enque(4)
# q.enque(7)
# print(q.deque())


##Intersection of two sorted array
def intersectTwoSorted(nums1, nums2):
    if len(nums1) == 0 or len(nums2) == 0:
        return []
    i = 0
    j = 0
    res = []
    while i < len(nums1) and j < len(nums2):
        if nums1[i] == nums2[j]:
            res.append(nums1[i])
            i += 1
            j += 1
        elif nums1[i] < nums2[j]:
            i += 1
        else:
            j += 1
    return res

# nums1 = [1,2,5, 8]
# nums2 = [2,5,9]
# print(intersectTwoSorted(nums1, nums2))

### ==============CAN SUM problem============

def canSum(nums, target):
    table = [False]*(target+1)
    table[0] = True
    for i in range(len(table)):
        if table[i] == True:
            for num in nums:
                if i+num <= len(table) - 1:
                    table[i+num] = True
    return table[-1]

# nums = [3, 8, 10, 2]
# target =
# print(canSum(nums, target))
##Time Complexity O(MN) space O(M)  M - targetSum n = length of nums


###======HOW SUM problem======================
def howSum(nums, target):
    table = [None]*(target+1)
    table[0] = []
    for i in range(len(table)):
        if table[i] != None:
            for num in nums:
                if i+num <= len(table) - 1:
                    table[i+num] = table[i] + [num]
    return table[-1]

# nums = [3, 5, 8]
# target = 13
# print(howSum(nums, target))
## Time O(M^2*N) spaceO(M^2)


###========BEST SUM problem===========================
def bestSum(nums, target):
    table = [None]*(target+1)
    table[0] = []
    res = []
    for i in range(len(table)):
        if table[i] != None:
            for num in nums:
                if i+num <= len(table) - 1:
                    combination = table[i] + [num]
                    if i+num == len(table) - 1:
                        res.append(combination)
                    if table[i+num] == None or len(table[i+num]) > len(combination):
                        table[i+num] = combination
    return res

# nums = [2, 4, 5, 6]
# target = 12
# print(bestSum(nums, target))
##Time O(m^2*N)  space O(M^2)

##=================================================================================

# Find average of each Sliding window. Given an input array of n elements and a sliding window fo size k,
# find the average of each sliding window.
# input = [1,2,3,4,5]
# k = 3
# output = [2.0, 3.0, 4.0]
# (1+2+3)/ 3 = 2.0
# (2+3+4)/3 = 3.0
# (3+4+5)/3 = 4.0

def avgWindow(nums, k):
    count = 0
    cSum = 0
    res = []
    left = 0
    for i in range(len(nums)):
        cSum += nums[i]
        count += 1
        if count == k:
            res.append(cSum/k)
            cSum -= nums[left]
            left += 1
            count -= 1
    return res

# nums = [2,8,9,0,6,0,5,7,3,4,9]
# k = 5
# print(avgWindow(nums, k))
