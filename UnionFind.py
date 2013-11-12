class UnionFind:
  def __init__(self, objects):
    self.parent = {}
    self.rank = {}
    self.num_clusters = 0
    # initialize which each node as its own parent
    for object in objects:
      self.parent[object] = object
      self.rank[object] = 0    
      self.num_clusters += 1
  
  def find(self, object):
    non_root = True
    path_nodes = []
    
    while non_root:
      above = self.parent[object]
      # if parent is a root, return parent
      if self.parent[above] == above:
        non_root = False
        root = above
      # otherwise, remember current node and move up to parent
      else:
        path_nodes.append(object)
        object = above
    
    # update all traversed nodes to point to root
    for node in path_nodes:
      self.parent[node] = root
      
    return root

  
  def union(self, object1, object2):
    # find both nodes
    s1 = self.find(object1)
    s2 = self.find(object2)
    if s1 != s2:
      self.num_clusters += -1
      # have root of lower rank point to root with higher rank
      if self.rank[s1] > self.rank[s2]:
        self.parent[s2] = s1
      elif self.rank[s2] > self.rank[s1]:
        self.parent[s1] = s2
      # if tie, have one root point to other root and adjust rank
      else:
        self.parent[s2] = s1
        self.rank[s1] += 1