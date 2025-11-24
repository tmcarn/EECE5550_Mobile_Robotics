import math

class OccGrid:
    def __init__(self, grid):
        self.occ_grid = grid

        self.dims = self.occ_grid.shape

        self.neighbor_map = {}
        self.node_set = self.build_node_set()

    def get_neighbors(self, cell):
        if cell in self.neighbor_map:
            return self.neighbor_map[cell]
        
        # Cell not yet registered in the map
        if self.occ_grid[*cell] == 0: # Not a free space, skip
            return None 
        
        row, col = cell
        neighbors = []
        for i in range(-1,2): # Row
            for j in range(-1,2): # Col
                if i==0 and j==0: 
                    continue # Skip cell itself
                else:
                    if (0 <= row+i <= self.dims[0]) and (0 <= col+j <= self.dims[1]): # Check bounds
                        neighbor = self.occ_grid[row+i, col+j]
                        if neighbor == 1: # Is FREE SPACE
                            neighbors.append((row+i, col+j))
        
        self.neighbor_map[cell] = neighbors # Cache for future lookup
        return neighbors
    
    def get_distance(self, cell1, cell2):
        y1, x1 = cell1
        y2, x2 = cell2

        return math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    
    def build_node_set(self):
        node_set = set()
        for row in range(self.dims[0]):
            for col in range(self.dims[1]):
                node_set.add((row, col))

        return node_set