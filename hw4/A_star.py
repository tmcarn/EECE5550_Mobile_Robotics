import numpy as np
from PIL import Image
import networkx as nx


class IMG2Graph:
    def __init__(self):
        self.img_path = "hw4/occupancy_map.png"
        occ_grid = self.load_image()

    def load_image(self):
        # Read image from disk using PIL
        occ_img = Image.open(self.img_path)

        # Convert to 0s and 1s
        occ_grid = (np.asarray(occ_img) > 0).astype(int)
        
        return occ_grid
    
    def build_graph(self):
        pass

    def check_neighbors(self):
        pass




# class AStar:
#     def __init__(self, V:set, start:int, goal:int, node_neighbors_func:function, weight_func:function, heuristic_func:function):
#         pass



graph = IMG2Graph()