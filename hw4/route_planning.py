from A_star import AStar
from occupancy_grid import OccGrid
from prob_road_map import PRM

from PIL import Image
from matplotlib import pyplot as plt

import numpy as np

def visualize_path(route, title, path):
    plt.figure(figsize=(10, 10))
    plt.imshow(occ_img, cmap='gray', origin='upper')

    # Extract path coordinates
    path_rows = [cell[0] for cell in route]
    path_cols = [cell[1] for cell in route]

    # Plot path
    plt.plot(path_cols, path_rows, 'r-', linewidth=2, label='Path')

    # Mark start and goal
    plt.plot(start[1], start[0], 'go', markersize=10, label='Start')
    plt.plot(goal[1], goal[0], 'bo', markersize=10, label='Goal')

    plt.legend()
    plt.title(f"{title}: (path length of {len(route)} nodes)")
    plt.axis("off")
    plt.savefig(path)
    plt.show()


img_path = "hw4/data/occupancy_map.png"

# Read image from disk using PIL
occ_img = np.asarray(Image.open(img_path))

# Convert to 0s and 1s
occ_grid = (occ_img > 0).astype(int)

occ = OccGrid(occ_grid)

start = (635, 140)
goal = (350, 400)

# Part 1: Route Planning with A* search
# astar = AStar(occ.node_set, start, goal, occ.get_neighbors, occ.get_distance, occ.get_distance)
# route = astar.run_search()
# visualize_path(route, "A* Path Planning", "hw4/plots/Astar.png")

# Part 2: Probabilistic Road Map
prm = PRM(occ, d_max=75, n_nodes=2_500)

prm.add_node(start)
prm.add_node(goal)

prm_astar = AStar(prm.G.nodes(), start, goal, prm.get_neighbors, occ.get_distance, occ.get_distance)
route = prm_astar.run_search()
visualize_path(route, "A* with PRM", "hw4/plots/PRM_Astar.png")

