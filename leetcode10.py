##1306. Jump Game III
"""Input: arr = [4,2,3,0,3,1,2], start = 5
Output: true
Explanation:
All possible ways to reach at index 3 with value 0 are:
index 5 -> index 4 -> index 1 -> index 3
index 5 -> index 6 -> index 4 -> index 1 -> index 3 """

def canReach(arr,start):
    n = len(arr)
    stack = [start]
    while len(stack) != 0:
        cur = stack.pop()
        if cur - arr[cur] >= 0:
            if arr[cur - arr[cur]] == 0:
                return True
            elif arr[cur - arr[cur]] > 0:
                stack.append(cur-arr[cur])
        if cur + arr[cur] < n:
            if arr[cur + arr[cur]] == 0:
                return True
            elif arr[cur + arr[cur]] > 0:
                stack.append(cur+arr[cur])
        arr[cur] = -1
    return False

print(canReach([3,0,2,1,2],2))

