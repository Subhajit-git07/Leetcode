##48. Rotate Image

def rotate(board):
    ##Transpose the matrix
    for row in range(len(board)):
        for col in range(row, len(board[0])):
            board[row][col], board[col][row] = board[col][row], board[row][col]

    ##reverse the row's
    n = len(board)
    #for row in range(len(board)):
       # board[row] = board[row][::-1]

    ##Reverse the col's
    for col in range(len(board)):
        for row in range(n//2):
            board[row][col], board[n-1-row][col] = board[n-1-row][col], board[row][col]

    return board

#90 degree rotation --> transpose + row reverse
#180 degree rotation --> row reverse + col reverse
#anti clockwise 90 degree --> transpose + col reverse

#board = [[1,2,3],[4,5,6],[7,8,9]]
#print(rotate(board))

##Group Anagrams

def groupAnagram(strs):
    dict1 = {}
    for s in strs:
        ch = [0]*26
        for c in s:
            pos = ord(c) - ord("a")
            ch[pos] += 1
        if tuple(ch) in dict1:
            dict1[tuple(ch)].append(s)
        else:
            dict1[tuple(ch)] = [s]
    return list(dict1.values())

#strs = ["eat","tea","tan","ate","nat","bat"]
#print(groupAnagram(strs))
##Time complexity O(nk) and space complexity O(n)

##Sqrt(x)

def mySqrt(x):
    left = 0
    right = x
    while left <= right:
        mid = (left+right)//2
        if mid*mid == x:
            result = mid
            break
        if mid*mid > x:
            right = mid - 1
        else:
            left = mid + 1
    return left

##print(mySqrt(9))

##inorder -- left root right
class TreeNode:
    def __init__(self,val):
        self.val = val
        self.left = None
        self.right = None

def inorder(root):
    if root is None:
        return None
    if root:
        if root.left:
            inorder(root.left)
        print(root.val)
        if root.right:
            inorder(root.right)

def inorder_iter(root):
    if root is None:
        return None
    stack = []
    output = []
    while root or stack:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        output.append(root.val)
        root = root.right
    return output

##check if tree is valid binary search tree

def validBST(root):
    return helper_bst(root, float('inf'), float('-inf'))

def helper(root, maxval, minval):
    if root is None:
        return True
    if root.val <= minval or root.val >= maxval:
        return False
    return helper(root.left, root.val, minval) and helper(root.right, maxval, root.val)
#Time complexity O(n) and space complexity O(d)

##If tree is symmetric or mirror of itself

def isSymmetric(root):
    return isMirror(root.left, root.right)

def isMirror(leftRoot, rightRoot):
    if leftRoot is None and rightRoot is None:
        return True
    if leftRoot is None or rightRoot is None:
        return False
    if leftRoot.val != rightRoot.val:
        return False
    return isMirror(leftRoot.left, rightRoot.right) and isMirror(leftRoot.right, rightRoot.left)


##Binary tree level order traversal

def levelOrder(root):
    stack = [[root]]
    res = [[root.val]]

    while len(stack) != 0:
        for node in stack:
            level = []
            levelVal = []
            if node.left:
                level.append(node.left)
                levelVal.append(node.left.val)
            if node.right:
                level.append(node.right)
                levelVal.append(node.right.val)

        if len(levelVal) != 0:
            res.append(levelVal)
        stack = level
    return res

##maximum depth of a binary tree

def maxDepth(root):
    if root is None:
        return 0
    return 1 + max(maxDepth(root.left), maxDepth(root.right))

##Binary Tree Level Order Traversal II

def levelOrderBottom(root):
    stack = [root]
    res = [[root.val]]

    while len(stack) != 0:
        level = []
        levelVal = []
        for node in stack:
            if node.left:
                level.append(node.left)
                levelVal.append(node.left.val)
            if node.right:
                level.append(node.right)
                levelVal.append(node.right.val)

        if len(levelVal) != 0:
            res.append(levelVal)
        stack = level
    res = res[::-1]
    return res

##Convert sorted array to binary search tree

def sortedArrayToBST(nums):
    if not nums:
        return None
    ind = len(nums) // 2
    root = nums[ind]
    root.left = sortedArrayToBST(nums[:ind])
    root.right = sortedArrayToBST((nums[ind+1:]))
    return root

##balanced binary tree

def isBalanced(root):
    return balance_helper(root) != -1

##Time complexity is O(n)

def balance_helper(root):
    if root is None:
        return 0
    leftH = balance_helper((root.left))
    rightH = balance_helper(root.right)

    if leftH == -1 or rightH == -1 or abs(leftH-rightH) > 1:
        return -1
    return 1 + max(leftH, rightH)

##Flatten binary tree
def flatten(root):
    if root is None:
        return None
    stack = [root]
    while len(stack) != 0:
        cur = stack.pop()
        if cur.right:
            stack.append(cur.right)
        if cur.left:
            stack.append(cur.left)

        cur.right = stack[-1]
        cur.left = None

##Populating next pointer in each node
class TreeNode:
    def __init__(self,val):
        self.val = val
        self.left = None
        self.right = None
        self.next =  None

root = TreeNode(0)

def connectNext(root):
    if root is None:
        return None
    if root.left:
        root.left.next = root.right
    if root.right:
        if root.next:
            root.right.next = root.next.left
    connectNext(root.left)
    connectNext(root.right)
    return root

##word break

def wordBreak(s, wordDict):
    memo = {}
    return wordBreak_helper(s, wordDict, memo)

def wordBreak_helper(s, wordDict, memo):
    if s == "":
        return True
    if s in memo:
        return memo[s]
    for str1 in wordDict:
        if s.startswith(str1):
            s1 = s[len(str1):]
            if wordBreak_helper(s1, wordDict, memo):
                memo[s1] = True
                return True
    memo[s] = False
    return False

##Binery Tree preorder traversal with iterative method

def preOrder(root):
    if root is None:
        return None
    output = []
    stack = [root]
    while len(stack) != 0:
        cur = stack.pop()
        output.append(cur.val)
        if cur.right:
            stack.append(cur.right)
        if cur.left:
            stack.append(cur.left)
    return output

def preOrder(root):
    if root is None:
        return None
    output = []
    stack = [root]
    while len(stack) != 0:
        cur = stack.pop()
        output.append(cur.val)
        if root.left:
            stack.append(cur.left)
        if root.right:
            stack.append(cur.right)
    return output[::-1]

##Binary tree right side view

def rightView(root):
    if root is None:
        return None
    output = []
    stack = [root]
    while len(stack) != 0:
        level = []
        output.append(stack[-1].val)
        for node in stack:
            if node.left:
                level.append(node.left)
            if node.right:
                level.append(node.right)

        stack = level
    return output

##kth largest element in array
##Use quickselect algo  and time complexity is O(n) and space complexity is O(1)
def kthLargest(nums, k):
    first_index = 0
    last_index = len(nums) - 1
    while first_index <= last_index:
        ind = partition(nums, first_index, last_index)
        if ind == k - 1:
            return nums[ind]
        if ind < k - 1:
            first_index = ind + 1
        else:
            last_index = ind - 1
    return None


def partition(nums, first_index, last_index):
    pivot_index = (first_index+last_index) // 2
    ##swap pivot and last element
    nums[pivot_index], nums[last_index] = nums[last_index], nums[pivot_index]

    for i in range(first_index, last_index):
        if nums[i] > nums[last_index]:
            nums[i], nums[first_index] = nums[first_index], nums[i]
            first_index += 1
    nums[first_index], nums[last_index] = nums[last_index], nums[first_index]
    return first_index

#nums = [3,9,7,4,1,20]
#print(kthLargest(nums, 6))

##Maximal square

def maximalSquare(matrix):
    table = [[0 for col in range(len(matrix[0])+1)] for row in range(len(matrix)+1)]
    m = len(table)
    n = len(table[0])
    height = 0
    for i in range(1,m):
        for j in range(1,n):
            if matrix[i-1][j-1] == "1":
                table[i][j] = 1 + min(table[i-1][j], table[i][j-1], table[i-1][j-1])
                height = max(height, table[i][j])
    return height*height

##227. Basic Calculator II
#Input: s = "3+2*2"
#Output: 7

def calculate(s):
    operation = "+"
    num = 0
    stack = []
    for i in range(len(s)):
        if s[i].isdigit():
            num = num*10 + int(s[i])
        if not s[i].isdigit() and s[i] in "+-*/" or i == len(s) - 1:
            if operation == "-":
                stack.append(-num)
            elif operation == "+":
                stack.append(num)
            elif operation == "*":
                stack.append(stack.pop()*num)
            elif operation == "/":
                stack.append(stack.pop()/num)
            operation = s[i]
            num = 0
    res = 0
    for j in stack:
        res += j
    return res

#s = "3+2*2"
#print(calculate(s))

##438. Find All Anagrams in a String
#Input: s = "cbaebabacd", p = "abc"
#Output: [0,6]

def findAnagrams(s, p):
    dict_p = {}
    for i in range(len(p)):
        if p[i] in dict_p:
            dict_p[p[i]] += 1
        else:
            dict_p[p[i]] = 1

    output = []
    dict_s = {}
    for i in range(len(s)):
        if s[i] in dict_s:
            dict_s[s[i]] += 1
        else:
            dict_s[s[i]] = 1

        if i >= len(p):
            if dict_s[s[i-len(p)]] == 1:
                del dict_s[s[i-len(p)]]
            else:
                dict_s[s[i-len(p)]] -= 1

        if dict_s == dict_p:
            output.append(i-len(p)+1)
    return output

#s = "cbaebabacd"
#p = "abc"
#print(findAnagrams(s, p))


##637. Average of Levels in Binary Tree

def averageOflevels(root):
    if root is None:
        return None
    stack = [root]
    res = [root.val]
    while len(stack) != 0:
        level = []
        cSum = 0
        count = 0
        for node in stack:
            if node.left:
                level.append(node.left)
                cSum += node.left.val
                count += 1
            if node.right:
                level.append(node.right)
                cSum += node.right.val
                count += 1

        if count != 0:
            res.append(cSum/count)
        stack = level
    return res

##429. N-ary Tree Level Order Traversal

def nArrayLevelOrder(root):
    if root is None:
        return None

    res = [root.val]
    stack = []
    while len(stack) != 0:
        level = []
        levelVal = []
        for node in stack:
            if node:
                for n in node.children:
                    level.append(n)
                    levelVal.append(n.val)
        if len(levelVal) != 0:
            res.append(levelVal)
        stack = level
    return res

##784. Letter Case Permutation
#Input: s = "a1b2"
#Output: ["a1b2","a1B2","A1b2","A1B2"]

def letterCasePermutation(s):
    queue = [s]

    for i in range(len(s)):
        if s[i].isalpha():
            size = len(queue)
            while size != 0:
                str1 = queue.pop()
                str1_left = str1[:i] + str1[i].upper() + str1[i+1:]
                str1_right = str1[:i] + str1[i].lower() + str1[i+1:]
                queue.insert(0, str1_left)
                queue.insert(0, str1_right)
                size -= 1
    return queue

#s = "a1b2"
#print(letterCasePermutation(s))

##1306. Jump Game III

#Input: arr = [4,2,3,0,3,1,2], start = 5
#Output: true

def canReach(nums, start):
    stack = [start]
    output = []
    while len(stack) != 0:
        cur = stack.pop()
        output.append(cur)
        if cur - nums[cur] >= 0:
            if nums[cur - nums[cur]] == 0:
                #return True
                output.append((cur - nums[cur]))
                return output
            elif nums[cur - nums[cur]] > 0:
                stack.append((cur-nums[cur]))

        if cur + nums[cur] < len(nums):
            if nums[cur + nums[cur]] == 0:
                #return True
                output.append((cur+nums[cur]))
                return output
            elif nums[cur + nums[cur]] > 0:
                stack.append((cur+nums[cur]))
        nums[cur] = - 1
    return False
#nums = [4,2,3,0,3,1,2]
#start = 0
#print(canReach(nums, start))

##1793. Maximum Score of a Good Subarray
#Input: nums = [1,4,3,7,4,5], k = 3
#Output: 15

def maximumScore(nums, k):
    ans = nums[k]
    left = k
    right = k
    minVal = nums[k]

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

        ans = max(ans, minVal*(right-left+1))
    return ans
#nums = [1,4,3,7,4,5]
#k = 3
#print(maximumScore(nums, k))
##Time comoplexity is O(n) and space complexity is O(1)

##109. Convert Sorted List to Binary Search Tree

def sortedListToBST(head):
    if head is None:
        return None
    if head.next is None:
        return TreeNode(head.val)
    slow = head
    fast = head.next
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next

    mid = slow.next
    slow.next = None
    root = TreeNode(mid)
    root.left = sortedListToBST(head)
    root.right = sortedListToBST(mid.next)
    return root
##Time complexity O(nlogn)

##113. Path Sum II

def pathSum2(root, targetSum):
    if root is None:
        return []
    res = []
    tempList = []
    helper_pathSum2(root, targetSum, tempList, res)
    return res

def helper_pathSum2(root, targetSum, tempList, res):
    if root is None:
        return 0
    if root.val == targetSum and root.left is None and root.right is None:
        res.append(tempList)
        return

    return helper_pathSum2(root.left, targetSum-root.val, tempList+[root.val], res) and helper_pathSum2(root.right,
                                                                targetSum-root.val, tempList+[root.val], res)

##Time complexity O(n) and space complexity O(1)


#125. Valid Palindrome

#Input: s = "A man, a plan, a canal: Panama"
#Output: true
#Explanation: "amanaplanacanalpanama" is a palindrome.

def isPelin(s):
    str1 = ''
    for c in s:
        if c.isalnum():
            str1 += c.lower()
    left = 0
    right = len(str1) - 1
    print(str1)
    while left <= right:
        if str1[left] != str1[right]:
            return False
        left += 1
        right -= 1
    return True

#s = "race a car"
#print(isPelin(s))

matrix = [[1,1,1,1,1],
          [1,1,0,0,1],
          [0,0,0,0,0],
          [0,0,0,1,1]]

def dfsTest(matrix, i, j):
    if i < 0 or i >= len(matrix) or j < 0 or j >= len(matrix[0]) or matrix[i][j] != 1:
        return
    matrix[i][j] = 2
    dfsTest(matrix, i+1, j)
    dfsTest(matrix, i-1, j)
    dfsTest(matrix, i, j+1)
    dfsTest(matrix, i, j-1)
count = 0
# for i in range(len(matrix)):
#     for j in range(len(matrix[0])):
#if matrix[i][j] == 1:
#dfsTest(matrix, 0, 0)

###============================================================================
##41. First Missing Positive
# Input: nums = [3,4,-1,1]
# Output: 2
#Algo- first convert the value of negative or 0 and greater then n to big number like n + 1
#      iterate through the values and make value - 1 index valu's to negative except grater then n values
##     again iterate and check the positive number index and return

def firstMissingPositive(nums):
    n = len(nums)
    for i in range(len(nums)):
        if nums[i] <= 0 or nums[i] > n:
            nums[i] = n + 1

    for i in range(len(nums)):
        cur = abs(nums[i])
        if cur > n :
            continue
        cur = cur - 1
        nums[cur] = -nums[cur]

    for i in range(len(nums)):
        if nums[i] > 0:
            return i + 1
    return n + 1
#nums = [7,8,9,11,12]
#print(firstMissingPositive(nums))

##PreOrder traversal

def preOrder(root):
    if root is None:
        return None
    stack = [root]

    while len(stack) != 0:
        cur = stack.pop()
        print(cur.val)
        if cur.right:
            stack.append(cur.right)
        if cur.left:
            stack.append(cur.left)

#left --> right --> root
def postOrder(root):
    if root is None:
        return None
    stack = [root]
    output = []
    while len(stack) != 0:
        cur = stack.pop()
        output.append(cur.val)
        if cur.left:
            stack.append(cur.left)
        if cur.right:
            stack.append(cur.right)

    output = output[::-1]
    return output


##33. Search in Rotated Sorted Array

#Input: nums = [4,5,6,7,0,1,2], target = 0
#Output: 4

def searchRotate(nums, target):
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
            if target <= nums[right] and target > nums[mid]:
                left = mid + 1
            else:
                right = mid - 1
    return -1
# nums = [1]
# target = 3
# print(searchRotate(nums, target))


##Min stack

class minStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val):
        self.stack.append(val)
        if len(self.min_stack) == 0:
            self.min_stack.append(val)
        else:
            if val < self.min_stack[-1]:
                self.min_stack.append(val)
            else:
                self.min_stack.append(self.min_stack[-1])

    def top(self):
        return self.stack[-1]

    def pop(self):
        self.stack.pop()
        self.min_stack.pop()

    def getMin(self):
        return self.min_stack[-1]

# mstack = minStack()
# mstack.push(3)
# mstack.push(-1)
# mstack.push(9)
# print(mstack.getMin())
# print(mstack.top())
# print(mstack.pop())
# print(mstack.getMin())
# print(mstack.top())


##239. Sliding Window Maximum

# Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
# Output: [3,3,5,5,6,7]

#Brute force

def slidingWindowMax(nums, k):
    output = []
    for i in range(len(nums)-k + 1):
        maxVal = max(nums[i:i+k])
        output.append(maxVal)

    return output

import collections
def slidingWindowMax2(nums,k):
    res = []
    r = 0
    deque = collections.deque()
    while r < len(nums):
        ##Chcek while sliding the window, if left being elements is maximum
        if r >= k and nums[r-k] == deque[0]:
            deque.popleft()
        ##Keep the queue monotonique non-decreasing
        while deque and deque[-1] < nums[r]:
            deque.pop()
        deque.append(nums[r])
        if r >= k - 1:
            res.append(deque[0])
        r += 1

    return res
# nums = [1,3,-1,-3,5,3,6,7]
# k = 3
# print(slidingWindowMax2(nums,k))


##Two Sum II - Input array is sorted

def twoSumSorted(nums, target):
    left = 0
    right = len(nums) - 1

    while left < right:
        if nums[left] + nums[right] == target:
            return [left+1, right+1]
        if nums[left] + nums[right] < target:
            left += 1
        else:
            right -= 1


# nums = [2, 7, 11, 15]
# target = 9
# print(twoSumSorted(nums, target))

##199. Binary Tree Right Side View

def RightSideView(root):
    if root is None:
        return None
    output = [root.val]
    stack = [root]

    while len(stack) != 0:
        level = []
        for node in stack:
            if node.left:
                level.append(node.left)
            if node.right:
                level.append(node.right)
        output.append(level[-1].val)
        stack = level
    return output

##116. Populating Next Right Pointers in Each Node

def populateNextRight(root):
    if root is None:
        return None
    if root.left:
        root.left.next = root.right
    if root.right:
        if root.next:
            root.right.next = root.next.left
    populateNextRight(root.left)
    populateNextRight(root.right)
    return root

#200. Number of Islands
def numberOfIsland(grid):
    m = len(grid)
    n = len(grid[0])
    count = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == "1":
                dfs_Island(grid, i, j)
                count += 1
    return count


##Time complexity O(mn) and space complexity O(1)
def dfs_Island(grid, i, j):
    if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]):
        return
    if grid[i][j] != "1":
        return
    grid[i][j] = "2"
    dfs_Island(grid, i+1, j)
    dfs_Island(grid, i-1, j)
    dfs_Island(grid, i, j+1)
    dfs_Island(grid, i, j-1)

# grid = [
#   ["1","1","0","0","0"],
#   ["1","1","0","0","0"],
#   ["0","0","1","0","0"],
#   ["0","0","0","1","1"]
# ]
#
# print(numberOfIsland(grid))

def getNext(number):
    Sum = 0
    while number > 0:
        rem = number % 10
        number = number // 10
        Sum += rem ** 2
    return Sum

##206. Reverse Linked List
class listNode:
    def __init__(self, val):
        self.val = val
        self.next = None

def reverseLinkedList(head):
    if head is None:
        return None
    prev = None
    while cur:
        temp = cur.next
        cur.next = prev
        prev = cur
        cur = temp
    return prev

##Time - O(n) and space O(1)

##234. Palindrome Linked List
##With using extra space

def pelindromLinkedList(head):
    if head is None:
        return None
    slow = head
    fast = head

    stack = []
    while fast and fast.next:
        stack.append(slow.val)
        slow = slow.next
        fast = fast.next.next

    if fast:
        slow = slow.next

    while slow and len(stack) != 0:
        if stack.pop() != slow.val:
            return False
        slow = slow.next
    return True


##9. Palindrome Number

def isPelindrom(num):
    output = 0
    bkp_num = num
    if num < 0:
        return False

    while x != 0:
        output = 10*output + x%10
        x = x// 10
    return bkp_num == output


class trieNode:
    def __init__(self):
        self.children = {}
        self.val = None
        self.next = None
        self.isEnd = False

class trie:
    def __init__(self):
        self.root = trieNode()

    def insert(self, word):
        parent = self.root
        for i, char in enumerate(word):
            if char not in parent.children:
                parent.children[char] = trieNode()
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
        for char in prefix:
            if char not in parent.children:
                return False
            parent = parent.children[char]
        return True

    def buildTrie(self, words):
        for word in words:
            self.insert(word)

# t = trie()
# words = ["google", "facebook", "face", "apple", "goo"]
# t.buildTrie(words)
# print(t.search("goo"))
# print(t.startsWith(""))


# 214. Shortest Palindrome
#
# Input: s = "aacecaaa"
# Output: "aaacecaaa"

##KMP algo

def shortestPelin(s):
    if len(s) == 0 or s == s[::-1]:
        return s
    revStr = s[::-1]
    newStr = s + "*" + revStr
    n = len(s)
    j = 0
    table = [0]*len(newStr)
    for i in range(1, len(newStr)):
        while j > 0 and newStr[i] != newStr[j]:
            j = table[j-1]

        if newStr[i] == newStr[j]:
            table[i] = j + 1
            j += 1
    res = revStr[:n-table[-1]] + s
    return res
# s = "abcd"
# print(shortestPelin(s))

##215. Kth Largest Element in an Array
##Using quick select algo

import random
def partition1(nums, first_index, last_index):
    pivot = random.randint(first_index, last_index)
    nums[pivot], nums[last_index] = nums[last_index], nums[pivot]

    for i in range(first_index, last_index):
        if nums[i] > nums[last_index]:
            nums[i], nums[first_index] = nums[first_index], nums[i]
            first_index += 1
    nums[first_index], nums[last_index] = nums[last_index], nums[first_index]
    return first_index

def kthLargest(nums, k):
    left = 0
    right = len(nums) - 1

    while left <= right:
        index = partition1(nums, left, right)
        if index == k - 1:
            return nums[index]
        if index <  k - 1:
            left = index + 1
        else:
            right = index - 1
    return None

# nums = [3,2,3,1,2,4,5,5,6]
# k = 4
# print(kthLargest(nums, k))

##973. K Closest Points to Origin

# Input: points = [[1,3],[-2,2]], k = 1
# Output: [[-2,2]]

def kClosestToOrigin(points, k):
    import math
    distance = []
    for p in points:
        distance.append((math.sqrt(p[1]**2 + p[0]**2), p))

    first = 0
    last = len(distance) - 1
    while first <= last:
        index = kClosestHelper(distance, first, last)
        if index == k - 1:
            break
        if index < k - 1:
            first = index + 1
        else:
            last = index - 1
    res = []
    for p in distance[:index+1]:
        res.append(p[1])
    return res

def kClosestHelper(distance, first, last):
    import random
    pivot = random.randint(first, last)
    distance[pivot], distance[last] = distance[last], distance[pivot]
    for i in range(first, last):
        if distance[i][0] < distance[last][0]:
            distance[first], distance[i] = distance[i], distance[first]
            first += 1
    distance[first], distance[last] = distance[last], distance[first]
    return first

# points = [[3,3],[5,-1],[-2,4]]
# k = 2
# print(kClosestToOrigin(points, k))

##692. Top K Frequent Words

# Input: words = ["i","love","leetcode","i","love","coding"], k = 2
# Output: ["i","love"]

def topKFreWords(words, k):
    wordDict = {}
    for word in words:
        if word in wordDict:
            wordDict[word] += 1
        else:
            wordDict[word] = 1
    nums = []
    for word in wordDict:
        nums.append([word, wordDict[word]])

    first = 0
    last = len(nums) - 1
    while first <= last:
        index = topKFreWords_Helper(nums, first, last)
        if index == k - 1:
            break
        if index < k - 1:
            first = index + 1
        else:
            last = index - 1
    res = []
    topK = nums[:index+1]
    topK.sort(key=lambda x:x[0])
    topK.sort(key=lambda x: x[1], reverse=True)
    print(topK)
    for i in topK:
        res.append(i[0])
    return res


def topKFreWords_Helper(nums, first, last):
    import random
    pivot = random.randint(first, last)
    nums[pivot], nums[last] = nums[last], nums[pivot]

    for i in range(first, last):
        if nums[i][1] > nums[last][1]:
            nums[i], nums[first] = nums[first], nums[i]
            first += 1
    nums[first], nums[last] = nums[last], nums[first]
    return first

# words = ["i","love","leetcode","i","love","coding"]
# k = 1
# print(topKFreWords(words, k))

##217. Contains Duplicate

#Input: nums = [1,2,3,1]
#Output: true

def containDuplicate(nums):
    Set = set()
    for num in nums:
        if num in Set:
            return True
        Set.add(num)
    return False
# nums = [1,1,1,3,3,4,3,2,4,2]
# print(containDuplicate(nums))

##242. Valid Anagram
def validAnagram(s1, s2):
    dictS1 = {}
    dictS2 = {}

    for i in range(len(s1)):
        if s1[i] in dictS1:
            dictS1[s1[i]] += 1
        else:
            dictS1[s1[i]] = 1

    for j in range(len(s2)):
        if s2[j] in dictS2:
            dictS2[s2[j]] += 1
        else:
            dictS2[s2[j]] = 1

    return dictS1 == dictS2
# s1 = "rat"
# s2 = "car"
#
# print(validAnagram(s1, s2))

##438. Find All Anagrams in a String
# Input: s = "cbaebabacd", p = "abc"
# Output: [0,6]

def findAllAnagrams(s, p):
    dictP = {}
    for char in p:
        if char in dictP:
            dictP[char] += 1
        else:
            dictP[char] = 1
    res = []
    output = []
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
                dictS[s[i - len(p)]] -= 1

        if dictP == dictS:
            res.append(i-len(p) + 1)
            output.append(s[i-len(p)+1:i+1])

    return output


# s = "cbaebabacd"
# p = "abc"
# print(findAllAnagrams(s, p))


##257. Binary Tree Paths
def binaryTreePath(root):
    if root is None:
        return []
    output = []
    helperTreePath(root, [str(root.val)], output)
    return output



def helperTreePath(root, res, output):
    if root.left is None and root.right is None:
        output.append("->".join(res))
        return
    if root.left:
        helperTreePath(root.left, res+[str(root.left.val)], output)
    if root.right:
        helperTreePath(root.right, res+[str(root.right.val)], output)


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
            num = num // 5
        else:
            return False
    return True

##268. Missing Number

def missingNum(nums):
    n = len(nums)
    allSum = (n*(n+1))//2
    cSum = sum(nums)
    return allSum - cSum

# nums = [9,6,4,2,3,5,7,0,1]
# print(missingNum(nums))

##136. Single Number
def singleNum(nums):
    setNum = set(nums)
    allSum = sum(set(nums))*2
    cSum = sum(nums)
    return allSum - cSum

# nums = [4,1,2,1,2]
# print(singleNum(nums))

##279. Perfect Squares

# Input: n = 12
# Output: 3
# Explanation: 12 = 4 + 4 + 4.

def perfectSquare(n):
    if n < 4:
        return n
    table = [0]*(n+1)

    table[1] = 1
    table[2] = 2
    table[3] = 3

    for i in range(4, n+1):
        table[i] = i
        j = 1
        while j*j <= i:
            table[i] = min(table[i], 1+table[i-j*j])
            j += 1
    return table[n]

# n = 13
# print(perfectSquare(n))

##35. Search Insert Position

def searchInsertPosition(nums, target):
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
# print(searchInsertPosition(nums, target))

##Valid parentheses

def validParen(s):
    stack = []
    for c in s:
        if c in "({[":
            stack.append(c)
        else:
            if len(stack) == 0:
                return False
            else:
                bottom = stack.pop()
                if not isValid(bottom,c):
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

# s = "[][][]((([)))"
# print(validParen(s))

def nextNode(lock):
    lockSet = set()
    lockArray = list(lock)
    for i in range(len(lockArray)):
        c = lockArray[i]
        if int(lockArray[i]) == 9:
            lockArray[i] = str(0)
        else:
            lockArray[i] = str(int(c) + 1)
        lockSet.add("".join(lockArray))
        lockArray[i] = c
        if int(lockArray[i]) == 0:
            lockArray[i] = str(9)
        else:
            lockArray[i] = str(int(c) - 1)
        lockSet.add("".join(lockArray))
        lockArray[i] = c
    return lockSet

# print(nextNode('0100'))

##Longest increasing subsequence

def binaryHelper(nums, num):
    left = 0
    right = len(nums) - 1
    while left <= right:
        mid = left + (right-left)//2
        if nums[mid] == num:
            return mid
        if nums[mid] < num:
            left = mid + 1
        else:
            right = mid - 1
    return left

def LIS(nums):
    table = [nums[0]]
    for i in range(1, len(nums)):
        if nums[i] > table[-1]:
            table.append(nums[i])
        else:
            idx = binaryHelper(table, nums[i])
            table[idx] = nums[i]
    return table

# nums = [7,7,7,7,7,7,7]
# print(LIS(nums))


#334. Increasing Triplet Subsequence
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


# nums = [5,4,3,2,1]
# print(increasingTriplet(nums))

##329. Longest Increasing Path in a Matrix
def longestIncreasingPath(grid):
    memo = {}
    maxLen = 0
    temp = []
    res = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            length = longestIncreasingDFS(grid, i, j, memo,temp, res)
            maxLen = max(maxLen, length)

    return maxLen

def longestIncreasingDFS(grid, i, j, memo, temp, res):
    length = 1
    if (i,j) in memo:
        return memo[(i,j)]
    temp.append(grid[i][j])
    dirs = [(0,-1), (0, 1), (1, 0), (-1, 0)]
    for dir in dirs:
        Row = i + dir[0]
        Col = j + dir[1]

        if Row >= 0 and Row < len(grid) and Col >= 0 and Col < len(grid[0]) and grid[Row][Col] > grid[i][j]:
            #length = max(length, longestIncreasingDFS(grid, Row, Col, memo) + 1)
            temp.append(grid[Row][Col])
            longestIncreasingDFS(grid, Row, Col, memo, temp, res)
            temp = []
    if len(res[0]) < len(temp):
        res[0] = temp
    memo[(i,j)] = res
    return res

# grid = [[9,9,4],[6,6,8],[2,1,1]]
# print(longestIncreasingPath(grid))

##337. House Robber III
def houseRobber3(root):
    if root is None:
        return 0
    memo = {}
    result = robber3Helper(root, memo)
    return result

def robber3Helper(root, memo):
    if root is None:
        return 0
    if root in memo:
        return memo[root]
    value = root.val
    if root.left:
        value += robber3Helper(root.left.left, memo) + robber3Helper(root.left.right, memo)
    if root.right:
        value += robber3Helper(root.right.left, memo) + robber3Helper(root.right.right, memo)
    val_without_root = robber3Helper(root.left, memo) + robber3Helper(root.right, memo)
    memo[root] = max(value, val_without_root)
    return memo[root]


##198. House Robber
# Input: nums = [1,2,3,1]
# Output: 4

def houseRobber(nums):
    if len(nums) == 0:
        return 0
    table = [0]*len(nums)
    if len(nums) >= 1:
        table[0] = nums[0]
    if len(nums) >= 2:
        table[1] = max(nums[0], nums[1])

    for i in range(2, len(nums)):
        table[i] = max(table[i-1], nums[i] + table[i-2])
    return table[-1]

# nums = [2, 5, 7, 10,30]
# print(houseRobber(nums))

##213. House Robber II
# Input: nums = [2,3,2]
# Output: 3

def houseRobber2(nums):
    n = len(nums)
    return max(houseRobber(nums[0:n-1]), houseRobber(nums[1:n]))
# nums = [2,3,2]
# print(houseRobber2(nums))

class ite:
    def __init__(self,nums):
        self.nums = nums
        self.cur = -1
        self.res = []
        self.iteratorHelper(nums, self.res)

    def iteratorHelper(self, nums, res):
        # if not nums:
        #     return
        for item in nums:
            if type(item) == int:
                self.res.append(item)
            else:
                self.iteratorHelper(item, self.res)

    def getList(self):
        return self.res

    def next(self):
        self.cur += 1
        return self.res[self.cur]

    def hasNext(self):
        return self.cur < len(self.res) - 1

    # def iterator(nums):
    #     res = []
    #     iteratorHelper(nums, res)
    #     return res
# nums = [[0,1],8,[[[1,2]]]]
# it = ite(nums)
# print(it.getList())
# print(it.next())
# print(it.next())
# print(it.next())
# print(it.hasNext())
# print(it.next())
# print(it.next())
# print(it.hasNext())


##344. Reverse String
# Input: s = ["h","e","l","l","o"]
# Output: ["o","l","l","e","h"]

def revStr(s):
    left = 0
    right = len(s) - 1
    while left <= right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1
    return s
# s = ["h","e","l","l","o"]
# print(revStr(s))


##451. Sort Characters By Frequency
# Input: s = "tree"
# Output: "eert"

def sortCharByFreq(s):
    sDict = {}
    for c in s:
        if c in sDict:
            sDict[c] += 1
        else:
            sDict[c] = 1

    res = ""
    sorted_sDict = sorted(sDict.items(), key=lambda x:x[1], reverse=True)
    for c in sorted_sDict:
        res += c[0]*c[1]
    return res

# s = "cccaaa"
# print(sortCharByFreq(s))

##1996. The Number of Weak Characters in the Game
# Input: properties = [[5,5],[6,3],[3,6]]
# Output: 0
# Explanation: No character has strictly greater attack and defense than the other.

def numberOfWeekCharacter(prop):
    prop.sort(key=lambda x:(x[0], -x[1]))
    maxDiff = 0
    res = 0
    for i in range(len(prop)-1, -1, -1):
        if prop[i][1] < maxDiff:

            res += 1
        maxDiff = max(maxDiff, prop[i][1])
    return res

# prop = [[1,5],[10,4],[4,3]]
# print(numberOfWeekCharacter(prop))

#380. Insert Delete GetRandom O(1)
class randomisedSet:
    import random
    def __init__(self):
        self.dict1 = {}
        self.array = []

    def insert(self, val):
        if val in self.dict1:
            return False
        self.dict1[val] = len(self.array)
        self.array.append(val)
        return True

    def remove(self, val):
        if val in self.dict1:
            last = self.array[-1]
            index_to_be_swaped = self.dict[val]
            self.array[index_to_be_swaped], self.dict[last] = last, index_to_be_swaped
            self.array.pop()
            del self.dict1[val]
            return True
        return False

    def getRandom(self):
        return random.choice(self.array)

# ran = randomisedSet()
# print(ran.insert(2))
# print(ran.insert(1))
# print(ran.insert(1))
# print(ran.getRandom())
# print(ran.remove(1))

##384. Shuffle an Array
class shuffle:
    import random
    def __init__(self, nums):
        self.nums = nums
        self.original = nums[:]

    def reset(self):
        return self.original

    def shuffleArray(self):
        res = self.nums

        for i in range(len(res)):
            swapIdx = random.randrange(i,len(res))
            res[i], res[swapIdx] = res[swapIdx], res[i]
        return res

# nums = [1,2,3,4,5]
# shu = shuffle(nums)
# print(shu.shuffleArray())

##387. First Unique Character in a String

# Input: s = "leetcode"
# Output: 0

def firstUiqueCharacter(s):
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

# s = "leetcode"
# print(firstUiqueCharacter(s))

##451. Sort Characters By Frequency

# Input: s = "tree"
# Output: "eert"
# Explanation: 'e' appears twice while 'r' and 't' both appear once.
# So 'e' must appear before both 'r' and 't'. Therefore "eetr" is also a valid answer.

def sortCharacterByFreq(s):
    dictS = {}
    for c in s:
        if c in dictS:
            dictS[c] += 1
        else:
            dictS[c] = 1
    res = ""

    d = sorted(dictS.items(), key=lambda x : x[1], reverse=True)
    for key, val in d:
        res += key*val
    return res

# s = "tree"
# print(sortCharacterByFreq(s))

##394. Decode String
# Input: s = "3[a]2[bc]"
# Output: "aaabcbc"

def decodeStr(s):
    curStr = ""
    count = 0
    stack = []
    for c in s:
        if c.isdigit():
            count = count*10 + int(c)
        elif c == "[":
            stack.append(curStr)
            stack.append(count)
            curStr = ""
            count = 0
        elif c == "]":
            curcount = stack.pop()
            prevStr = stack.pop()
            curStr = prevStr + curStr*curcount
        else:
            curStr += c

    return curStr

# s = "3[a]2[bc]"
# print(decodeStr(s))

def RGB(array):
    left = 0
    right = len(array) - 1
    cur = 0
    while cur <= right:
        if array[cur] == "R":
            array[cur], array[left] = array[left], array[cur]
            cur += 1
            left += 1
        elif array[cur] == "B":
            array[cur], array[right] = array[right], array[cur]
            right -= 1
        else:
            cur += 1
    return array

# array = ['G', 'B', 'R', 'R', 'B', 'R', 'G']
# print(RGB(array))

#412. Fizz Buzz
# Input: n = 3
# Output: ["1","2","Fizz"]

def fizzBuzz(n):
    res = []
    for i in range(1, n+1):
        if i%3 == 0 and i%5 == 0:
            i = "FizzBuzz"
        elif i%3 == 0:
            i = "Fizz"
        elif i%5 == 0:
            i = "Buzz"
        else:
            i = str(i)
        res.append(i)
    return res
# n = 5
# print(fizzBuzz(n))


##5. Longest Palindromic Substring

def longestPelinSubStr(s):
    res = ""
    for i in range(len(s)):
        oddStr = longestPelinSubStr_helper(i, i, s)
        if len(res) < len(oddStr):
            res = oddStr
        evenStr = longestPelinSubStr_helper(i, i+1, s)
        if len(res) < len(evenStr):
            res = evenStr
    return res

def longestPelinSubStr_helper(left, right, s):
    while left >= 0 and right < len(s) and s[left] == s[right]:
        left -= 1
        right += 1
    return s[left+1:right]

# s = "ac"
# print(longestPelinSubStr(s))
##Time complexity O(n^2) and space complexity O(1)

##11. Container With Most Water
# Input: height = [1,8,6,2,5,4,8,3,7]
# Output: 49

def containerWithMostWater(height):
    maxArea = 0
    left = 0
    right = len(height) - 1

    while left < right:
        dis = right - left
        maxArea = max(maxArea, dis*(min(height[left], height[right])))
        if height[left] <= height[right]:
            left += 1
        else:
            right -= 1

    return maxArea

# height = [1,2,1]
# print(containerWithMostWater(height))
##Time O(n) and space O(1)


# For example, given the list of flights [('SFO', 'HKO'), ('YYZ', 'SFO'), ('YUL', 'YYZ'), ('HKO', 'ORD')] and
# starting airport 'YUL', you should return the list ['YUL', 'YYZ', 'SFO', 'HKO', 'ORD'].

def airport(edges, start):
    adj_list = {}
    for u, v in edges:
        adj_list[u] = []
        adj_list[v] = []
    for u, v in edges:
        adj_list[u].append(v)
    stack = [start]
    visited = set()
    res = []
    print(adj_list)
    while stack:
        node = stack.pop()
        visited.add(node)
        for nei in adj_list[node]:
            stack.append(nei)
        res.append(node)
    return res


# edges = [('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'A')]
# print(airport(edges, 'A'))

# 417. Pacific Atlantic Water Flow

def pacificAtlantic(heights):
    if len(heights) == 0 or len(heights[0]) == 0:
        return []
    m = len(heights)
    n = len(heights[0])
    atlantic = [[False for col in range(n)] for row in range(m)]
    pacific = [[False for col in range(n)] for row in range(m)]
    prev = float('-inf')
    res = []
    for i in range(m):
        pacificAtlantic_DFS(heights, i, 0, prev, pacific)
        pacificAtlantic_DFS(heights, i, n-1, prev, atlantic)

    for i in range(n):
        pacificAtlantic_DFS(heights, 0, i, prev, pacific)
        pacificAtlantic_DFS(heights, m-1, i, prev, atlantic)

    for i in range(m):
        for j in range(n):
            if atlantic[i][j] and pacific[i][j]:
                res.append([i,j])
    return res



def pacificAtlantic_DFS(heights, i, j, prev, ocean):
    if i < 0 or i >= len(heights) or j < 0 or j >= len(heights[0]):
        return
    if heights[i][j] < prev or ocean[i][j]:
        return
    ocean[i][j] = True

    pacificAtlantic_DFS(heights, i+1, j, heights[i][j], ocean)
    pacificAtlantic_DFS(heights, i-1, j, heights[i][j], ocean)
    pacificAtlantic_DFS(heights, i, j+1, heights[i][j], ocean)
    pacificAtlantic_DFS(heights, i, j-1, heights[i][j], ocean)

# heights = [[2,1],[1,2]]
# print(pacificAtlantic(heights))

# 433. Minimum Genetic Mutation

# Input: start = "AACCGGTT", end = "AAACGGTA", bank = ["AACCGGTA","AACCGCTA","AAACGGTA"]
# Output: 2

def minGeneticMutation(start, end, bank):
    if len(end) == 0 or len(bank) == 0 or len(start) == 0:
        return -1
    if end not in bank:
        return -1

    bank = set(bank)
    stack = [start]
    visited = set()
    level = 0
    while len(stack) != 0:
        cur = stack.pop()
        if cur == end:
            return level
        visited.add(cur)
        tempLevel = []
        for i in range(len(cur)):
            for ch in "ACGT":
                newStr = cur[:i] + ch + cur[i+1:]
                if newStr not in visited and newStr in bank:
                    tempLevel.append(newStr)
        if len(stack) == 0:
            stack = tempLevel
            level += 1
    return -1

# start = "AACCGGTT"
# end = "AACCGGTA"
# bank = ["AACCGGTA"]
# print(minGeneticMutation(start, end, bank))

# 127. Word Ladder

# Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
# Output: 5
# Explanation: One shortest transformation
# sequence is "hit" -> "hot" -> "dot" -> "dog" -> cog", which is 5 words long.

def wordLadder(beginWord, endWord, wordList):
    if len(beginWord) == 0 or len(endWord) == 0 or len(wordList) == 0:
        return 0
    if endWord not in wordList:
        return 0
    wordList = set(wordList)
    stack = [beginWord]
    level = 1
    visited = set()
    alpha = "abcdefghijklmnopqrstuvwxyz"
    while len(stack) != 0:
        size = len(stack)
        while size != 0:
            cur = stack.pop()
            if cur == endWord:
                return level
            visited.add(cur)
            tempLevel = []
            for i in range(len(cur)):
                for ch in alpha:
                    newStr = cur[:i] + ch + cur[i+1:]
                    if newStr not in visited and newStr in wordList:
                        tempLevel.append(newStr)
            size -= 1
        stack = tempLevel
        level += 1
    return 0

# beginWord = "hit"
# endWord = "cog"
# wordList = ["hot","dot","dog","lot","log","cog"]
# print(wordLadder(beginWord, endWord, wordList))


# 438. Find All Anagrams in a String
# Input: s = "cbaebabacd", p = "abc"
# Output: [0,6]
# Explanation:
# The substring with start index = 0 is "cba", which is an anagram of "abc".
# The substring with start index = 6 is "bac", which is an anagram of "abc".

def allAnagramInStr(s, p):
    dict_p = {}
    for c in p:
        if c in dict_p:
            dict_p[c] += 1
        else:
            dict_p[c] = 1

    dict_s = {}
    output = []
    for i in range(len(s)):
        if s[i] in dict_s:
            dict_s[s[i]] += 1
        else:
            dict_s[s[i]] = 1

        if i >= len(p):
            if dict_s[s[i-len(p)]] == 1:
                del dict_s[s[i-len(p)]]
            else:
                dict_s[s[i-len(p)]] -= 1

        if dict_p == dict_s:
            output.append(i-len(p)+1)

    return output

# s = "cbaebabacd"
# p = "abc"
# print(allAnagramInStr(s, p))

##448. Find All Numbers Disappeared in an Array
# Input: nums = [4,3,2,7,8,2,3,1]
# Output: [5,6]

def disappearedInArray(nums):
    missing = []
    for num in nums:
        pos = abs(num) - 1
        if nums[pos] > 0:
            nums[pos] *= -1
    for i in range(len(nums)):
        if nums[i] > 0:
            missing.append(i+1)
    return missing

# nums = [4,3,2,7,8,2,3,1]
# print(disappearedInArray(nums))

#449. Serialize and Deserialize BST

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def serializeBST(root):
    res = []
    serializeBST_Helper(root, res)
    return "".join(res)

def serializeBST_Helper(root, res):
    if root is None:
        return
    res.append(str(root.val)+",")
    serializeBST_Helper(root.left)
    serializeBST_Helper(root.right)

##Serialize is O(n) time complexity

def deserializeBST(data):
    data = data.split(',')
    data.pop()
    root = None
    for n in data:
        n = int(n)
        node = TreeNode(n)
        stack = []
        if root is None:
            root = node
            stack.append(node)
        else:
            if stack[-1].val > n:
                stack[-1].left = node
            else:
                while stack and stack[-1].val < n:
                    u = stack.pop()
                u.right = node
            stack.append(node)
    return root

##Deserialize is O(n) time complexity


#451. Sort Characters By Frequency

# Input: s = "tree"
# Output: "eert"
# Explanation: 'e' appears twice while 'r' and 't' both appear once.
# So 'e' must appear before both 'r' and 't'. Therefore "eetr" is also a valid answer.

def sortCharByFreq(s):
    dict_s = {}
    for char in s:
        if char in dict_s:
            dict_s[char] += 1
        else:
            dict_s[char] = 1

    sortedDict = sorted(dict_s.items(), key=lambda x : x[1], reverse=True)
    res = ""
    for i in sortedDict:
        res += i[0]*i[1]
    return res

# s = "tree"
# print(sortCharByFreq(s))

#453. Minimum Moves to Equal Array Elements
# Input: nums = [1,2,3]
# Output: 3
# Explanation: Only three moves are needed (remember each move increments two elements):
# [1,2,3]  =>  [2,3,3]  =>  [3,4,3]  =>  [4,4,4]

def minMove(nums):
    minNum = min(nums)
    steps = 0
    for num in nums:
        steps += num - minNum
    return steps

# nums = [1, 2 ,3]
# print(minMove(nums))


##Minimum Moves to Equal Array Elements II
# Input: nums = [1,2,3]
# Output: 2
# Explanation:
# Only two moves are needed (remember each move increments or decrements one element):
# [1,2,3]  =>  [2,2,3]  =>  [2,2,2]

def minMove2(nums):
    left = 0
    right = len(nums) - 1
    steps = 0
    nums.sort()
    while left < right:
        steps += nums[right] - nums[left]
        left += 1
        right -= 1

    return steps
#Time complexity O(nlogn) and space complexity O(1)
# nums = [1,2, 3]
# print(minMove2(nums))

##463. Island Perimeter
def islandPerimeter(grid):
    peri = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                peri += 4
                if i > 0:
                    peri -= grid[i-1][j]
                if j > 0:
                    peri -= grid[i][j-1]
                if i < len(grid) - 1:
                    peri -= grid[i+1][j]
                if j < len(grid) - 1:
                    peri -= grid[i][j+1]

    return peri

# grid = [[1,0]]
# print(islandPerimeter(grid))

##695. Max Area of Island
def maxAreaOfIsland(grid):
    maxArea = 0
    visited = [[False for col in range(len(grid[0]))] for row in range(len(grid))]
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            maxArea = max(maxArea, dfs_maxIsland(grid, i, j, visited))
    return maxArea


def dfs_maxIsland(grid, i, j, visited):
    if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] == 0:
        return 0
    if visited[i][j]:
        return 0
    visited[i][j] = True
    count = 1
    count += dfs_maxIsland(grid, i+1, j, visited)
    count += dfs_maxIsland(grid, i-1, j, visited)
    count += dfs_maxIsland(grid, i, j+1, visited)
    count += dfs_maxIsland(grid, i, j-1, visited)
    return count

# grid = [[0,0,1,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,1,1,0,1,0,0,0,0,0,0,0,0],[0,1,0,0,1,1,0,0,1,0,1,0,0],[0,1,0,0,1,1,0,0,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,1,1,0,0,0,0]]
# print(maxAreaOfIsland(grid))


##200. Number of Islands
def numOfLand(grid):
    if not grid:
        return 0
    visited = [[False for col in range(len(grid[0]))] for row in range(len(grid))]
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == "1":
                numOfLandDFS(grid, i, j, visited)
                count += 1
    return count



def numOfLandDFS(grid, i, j, visited):
    if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] != "1":
        return
    if visited[i][j]:
        return
    visited[i][j] = True
    grid[i][j] = "2"

    numOfLandDFS(grid, i+1, j, visited)
    numOfLandDFS(grid, i-1, j, visited)
    numOfLandDFS(grid, i, j+1, visited)
    numOfLandDFS(grid, i, j-1, visited)


# grid = [
#   ["1","1","0","0","0"],
#   ["1","1","0","0","0"],
#   ["0","0","1","0","0"],
#   ["0","0","0","1","1"]
# ]
#
# print(numOfLand(grid))