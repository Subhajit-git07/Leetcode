##48. Rotate Image 90 degree

def rotate(matrix):
    #for row in range(len(matrix)):
        #for col in range(row, len(matrix[0])):
            #matrix[row][col], matrix[col][row] = matrix[col][row], matrix[row][col]
    n = len(matrix)
    for row in range(len(matrix)):
        matrix[row] = matrix[row][::-1]

    for col in range(len(matrix)):
        for row in range(n//2):
            matrix[row][col], matrix[n-1-row][col] = matrix[n-1-row][col], matrix[row][col]
    return matrix


#matrix = [[1,2,3],[4,5,6],[7,8,9]]
#print(rotate(matrix))

'''
1 2 3    7 4 1  9 8 7
4 5 6    8 5 2  6 5 4
7 8 9    9 6 3  3 2 1
'''


def topKFrequent(nums, k):
    nums_dict = {}
    for num in nums:
        if num in nums_dict:
            nums_dict[num] += 1
        else:
            nums_dict[num] = 1
    keys = list(nums_dict.keys())

    start_index = 0
    last_index = len(keys) - 1

    while start_index <= last_index:
        index = partition(keys, start_index, last_index, nums_dict)
        if index == k - 1:
            return keys[:index + 1]
        elif index < k - 1:
            start_index = index + 1
        else:
            last_index = index - 1


def partition(nums, start_index, last_index, nums_dict):
    pivot = (start_index + last_index) // 2
    # swap pivot with last
    nums[pivot], nums[last_index] = nums[last_index], nums[pivot]

    for i in range(start_index, last_index):
        if nums_dict[nums[i]] > nums_dict[nums[last_index]]:
            nums[i], nums[start_index] = nums[start_index], nums[i]
            start_index += 1
    # swap start and last
    nums[start_index], nums[last_index] = nums[last_index], nums[start_index]
    return start_index

'''
import math
d_list = []
points = [[0,1],[1,0]]
for p in points:
    dist = math.sqrt(p[1]**2 + p[0]**2)
    d_list.append((dist,p))

distances = [[point[0]**2 + point[1]**2, point] for point in points]
print(distances)
'''

words = ["i", "love", "leetcode", "i", "love", "coding"]
words_dict = {}
for w in words:
    if w in words_dict:
        words_dict[w] += 1
    else:
        words_dict[w] = 1


'''
k = 2
heap = [(-freq, word) for word, freq in words_dict.items()]

out = heap
out.sort()
out = heap[:3]
a = [word for _, word in out]
print(a)
'''

##Jump game

def canJump(nums):
    maxReach = 0
    for i in range(len(nums)):
        if i > maxReach:
            return False
        maxReach = max(maxReach, i+nums[i])
    return True

#nums = [3,2,1,0,4]
#print(canJump(nums))

#1306. Jump Game III

#Input: arr = [4,2,3,0,3,1,2], start = 5
#Output: true

def jumpGame3(nums, start):
    stack = [start]
    level = []
    while len(stack) != 0:
        cur = stack.pop()
        level.append(cur)
        if cur - nums[cur] >= 0:
            if nums[cur - nums[cur]] == 0:
                level.append(cur - nums[cur])
                return level
            elif nums[cur - nums[cur]] > 0:
                stack.append(cur - nums[cur])
        if cur + nums[cur] < len(nums):
            if nums[cur + nums[cur]] == 0:
                level.append(cur + nums[cur])
                return level
            elif nums[cur + nums[cur]] > 0:
                stack.append(cur + nums[cur])
        nums[cur] = -1
    return False
nums = [3,0,2,1,2]
start = 2
#print(jumpGame3(nums, start))

def jumpGame3_paths(nums, start):
    stack = [start]
    level = [start]
    result = []
    helper_jumpGame3(nums, stack, start, level, result)
    return result

def helper_jumpGame3(nums, stack, start, level, result):
    while len(stack) != 0:
        cur = stack.pop()
        if cur - nums[cur] >= 0:
            if nums[cur - nums[cur]] == 0:
                level.append(cur - nums[cur])
                result.append(level)
                return
            elif nums[cur - nums[cur]] > 0:
                helper_jumpGame3(nums, stack+[cur - nums[cur]], start, level+[cur - nums[cur]], result)
        if cur + nums[cur] < len(nums):
            if nums[cur + nums[cur]] == 0:
                level.append(cur + nums[cur])
                result.append(level)
                return
            elif nums[cur + nums[cur]] > 0:
                helper_jumpGame3(nums, stack + [cur + nums[cur]], start, level + [cur + nums[cur]], result)
        nums[cur] = -1
    return False

##print(jumpGame3_paths(nums, start))


##139. Word Break

#Input: s = "applepenapple", wordDict = ["apple","pen"]
#Output: true

def wordBreak(s, wordDict):
  memo = {}
  return helper_wb(s, wordDict, memo)

def helper_wb(s, wordDict, memo):
    if s == "":
        return True
    if s in memo:
        return memo[s]
    for str1 in wordDict:
        if s.startswith(str1):
            s1 = s[len(str1):]
            if helper_wb(s1, wordDict, memo):
                memo[s1] = True
                return True
    memo[s] = False
    return False

#s = "catsandog"
#wordDict = ["cats","dog","sand","and","cat"]
#print(wordBreak(s, wordDict))

##Trie implementation

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.child = {}
        self.endhere = False

class trie:
    def __init__(self):
        self.root = TreeNode(None)

    def insert(self, word):
        parent = self.root
        for i,char in enumerate(word):
            if char not in parent.child:
                parent.child[char] = TreeNode(char)
            parent = parent.child[char]
            if i == len(word) - 1:
                parent.endhere = True

    def build_trie(self, words):
        for word in words:
            self.insert(word)

    def insert1(self, word):
        node = self.root
        for char in word:
            if char not in node.child:
                node.child[char] = TreeNode()
            node = node.child[char]
        node.endhere = True

    def search(self, word):
        parent = self.root
        for char in word:
            if char not in parent.child:
                return False
            parent = parent.child[char]
        return parent.endhere

    def search1(self, word):
        node = self.root
        for char in word:
            if char in node.child:
                node = node.child[char]
            else:
                return False

        return node.endhere

    def startsWith(self, prefix):
        parent = self.root
        for char in prefix:
            if char not in parent.child:
                return False
            parent = parent.child[char]
        return True

    def print_t(self):
        for i in self.root.child:
            print(self.root.child[i].val)

    def autoComplete(self, partial_word):
        parent = self.root
        word_list = []
        for char in partial_word:
            if char in parent.child:
                parent = parent.child[char]
            else:
                return word_list
        #word_list.append(partial_word)
        self.walk_trie(parent, partial_word, word_list)
        return word_list

    def walk_trie(self, parent, partial_word, word_list):
        if parent.child:
            for char in parent.child:
                word_new = partial_word + char
                if parent.child[char].endhere:
                    word_list.append(word_new)
                self.walk_trie(parent.child[char], word_new, word_list)

import nltk
#nltk.download()
# from nltk.corpus import words
# word_list = words.words()
# # prints 236736
# print(len(word_list))

t = trie()

words = ["cook", "hook", "ram", "rat", "raaaa", "ratios", "hover",]
t.build_trie(words)
print(t.search("ra"))
print(t.print_t())
print(t.autoComplete("ra"))

