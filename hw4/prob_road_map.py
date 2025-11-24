import numpy as np
from skimage.draw import line
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

from occupancy_grid import OccGrid

class PRM:
    def __init__(self, grid:OccGrid, d_max, n_nodes):
        self.grid = grid
        self.grid_data = grid.occ_grid
        self.G = nx.Graph()

        self.d_max = d_max
        self.n_nodes = n_nodes

        self.build_map()
        self.viz_graph()

    def sample_grid(self):
        random_pixel = 0
        height, width = self.grid_data.shape
        while random_pixel <= 0: # While invalid, keep picking
            random_row = np.random.randint(0, height)
            random_col = np.random.randint(0, width)
            random_pixel = self.grid_data[random_row, random_col]

        return (random_row, random_col)
    
    def is_locally_traversable(self, cell1, cell2):
        r_list, c_list = line(*cell1, *cell2) # Points on the line between cell1 and cell2

        # All cells along line must be free
        return np.all(self.grid_data[r_list, c_list] == 1)
    
    def add_node(self, new_node):
        self.G.add_node(new_node, pos=new_node) # Add to graph
        
        for node in self.G.nodes():
            distance = self.grid.get_distance(node, new_node)
            if (node != new_node) and (distance <= self.d_max) and (self.is_locally_traversable(node, new_node)):
                self.G.add_edge(node, new_node, weight=distance)

    def build_map(self):
        for i in tqdm(range(self.n_nodes)):
            node = self.sample_grid()
            self.add_node(node)

    def get_neighbors(self, node):
        neighbor_list = list(self.G[node])
        return neighbor_list

    def viz_graph(self):

        # Visualize PRM on occupancy grid
        fig, ax = plt.subplots(figsize=(12, 12))

        # Show occupancy grid
        ax.imshow(self.grid_data, cmap='gray', origin='upper')

        # Get positions (swap to x=col, y=row for plotting)
        pos = {node: (node[1], node[0]) for node in self.G.nodes()}

        # Draw graph on top of image
        nx.draw_networkx_nodes(self.G, pos, 
                            node_color='tab:orange', 
                            node_size=20,
                            alpha=0.7,
                            ax=ax)

        nx.draw_networkx_edges(self.G, pos, 
                            edge_color='tab:blue', 
                            width=0.5,
                            alpha=0.1,
                            ax=ax)

        plt.title('PRM Roadmap')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("hw4/plots/PRM.png")
        plt.show()
                