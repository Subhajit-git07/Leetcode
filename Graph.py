# class graph:
#     def __init__(self, nodes, isDirected=False):
#         self.isDirected = isDirected
#         self.nodes = nodes
#         self.adj_list = {}
#
#         for node in self.nodes:
#             self.adj_list[node] = []
#
#     def add_edge(self, u, v):
#         self.adj_list[u].append(v)
#         if not self.isDirected:
#             self.adj_list[v].append(u)
#
#     def printGraph(self):
#         for node in self.nodes:
#             print(node,"--->",self.adj_list[node])
#
#     def degree(self, node):
#         return len(self.adj_list[node])
'''
nodes = ["A","B","C","D","E"]
all_edges = [("A","B"),("A","C"),("B","D"),("C","D"),("C","E"),("D","E")]
g = graph(nodes,isDirected=True)
'''
'''
for u,v in all_edges:
    g.add_edge(u,v)
print(g.printGraph())
'''

##Depth first search
##DFS
#     0
#    / \
#   1   2
#  /    /\
# 5    3  4

# vertexList = [0,1,2,3,4,5]
# edgeList = [(0,1),(0,2),(1,5),(2,3),(2,4)]
# graph = (vertexList,edgeList)

# vertexList = [0,1,2,3]
# edgeList = [[1,0],[2,0],[3,1],[3,2]]
# graph = (vertexList,edgeList)

def graphDFS(graph, start):
    nodes, edgeList = graph
    adj_list = {}
    for node in nodes:
        adj_list[node] = []
    for u,v in edgeList:
        adj_list[u].append(v)

    output = []
    stack = [start]
    while len(stack) != 0:
        cur = stack.pop()
        for neighbour in adj_list[cur]:
            if neighbour not in output:
                stack.append(neighbour)
        output.append(cur)
    return output

# print(graphDFS(graph, 0))

def graphBFS(graph, start):
    nodes, edgeList = graph
    adj_list = {}
    for node in nodes:
        adj_list[node] = []
    for u,v in edgeList:
        adj_list[u].append(v)

    queue = [start]
    output = []
    while len(queue) != 0:
        cur = queue.pop()
        for neighbour in adj_list[cur]:
            if neighbour not in output:
                queue.insert(0, neighbour)
        output.append(cur)
    return output

#print(graphBFS(graph, 0))

#     0
#    / \
#   1   2
#  /    /\
# 5    3  4

# vertexList = [0,1,2,3,4,5]
# edgeList = [(0,1),(1,2),(2,0),(0,2),(1,5),(2,3),(2,4)]
# graph = (vertexList,edgeList)

##Has path
def buildList(graph):
    adj_list = {}
    nodes, edgeList = graph
    for node in nodes:
        adj_list[node] = []
    for u, v in edgeList:
        adj_list[u].append(v)

    return adj_list

def hasPath(graph, start, end):
    adj_list = buildList(graph)
    stack = [start]
    while len(stack) != 0:
        cur = stack.pop()
        if cur == end:
            return True
        for neighbour in adj_list[cur]:
            stack.append(neighbour)
    return False

#print(hasPath(graph, 3, 2))

##Undirected

#     0
#    / \
#   1---2
#  /    /\
# 5    3  4

# vertexList = [0,1,2,3,4,5]
# edgeList = [(0,1),(1,2),(2,0),(1,5),(2,3),(2,4)]
# graph = (vertexList,edgeList)

##Has path
def buildListU(graph):
    adj_list = {}
    nodes, edgeList = graph
    for node in nodes:
        adj_list[node] = []
    for u, v in edgeList:
        adj_list[u].append(v)
        adj_list[v].append(u)

    return adj_list

def hasPathUn(graph, start, end):
    adj_list = buildListU(graph)
    #print(adj_list)
    stack = [start]
    visited = set()
    while len(stack) != 0:
        print(stack)
        cur = stack.pop()
        if cur in visited:
            return "cycle detected"
        visited.add(cur)
        if cur == end:
            return True
        for neighbour in adj_list[cur]:
            if neighbour not in visited:
                stack.append(neighbour)
    return False

# print(hasPathUn(graph, 0, 4))

##Largest component

#adj_list1 = {0:[8,1,5], 1:[0], 2:[3,4], 3:[2,4], 4:[3,2], 5:[0,8], 8:[0,5]}
def largestComponent(adj_list):
    nodes = list(adj_list1.keys())
    visited = set()
    maxCount = 0
    for node in nodes:
        if node not in visited:
            count = largestComponentDFS(adj_list1, node, visited)
            maxCount = max(maxCount, count)
    return count

def largestComponentDFS(adj_list1, node, visited):
    #visited = set()
    count = 0
    stack = [node]
    while len(stack) != 0:
        cur = stack.pop()
        visited.add(cur)
        count += 1
        for neighbour in adj_list1[cur]:
            if neighbour not in visited:
                stack.append(neighbour)
    return count

# print(largestComponent(adj_list1))