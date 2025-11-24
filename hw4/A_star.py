import numpy as np
import math
from typing import Callable
import heapq



class AStar:
    def __init__(self, 
                 V:set, 
                 start:tuple, 
                 goal:tuple, 
                 node_neighbors_func:Callable, 
                 weight_func:Callable, 
                 heuristic_func:Callable):
        
        self.node_set = V
        self.cost_to = {}
        self.est_total_cost = {}
        self.set_init_cost_maps()

        self.predecessors = {}

        self.start = start
        self.goal = goal

        self.neighbors = node_neighbors_func
        self.heuristic = heuristic_func
        self.weight = weight_func

        self.cost_to[self.start] = 0
        distance_to_goal = self.heuristic(self.start, self.goal)
        self.est_total_cost[self.start] = distance_to_goal

        self.pq = []

        # Add Start Node to PQ
        heapq.heappush(self.pq, (distance_to_goal, start)) # (priority, item)

    def set_init_cost_maps(self):
        for node in self.node_set:
            self.cost_to[node] = math.inf
            self.est_total_cost[node] = math.inf

    def run_search(self):
        while len(self.pq) > 0:
            curr_cell = heapq.heappop(self.pq)[1]
            if curr_cell == self.goal:
                return self.recover_path()# Shortest Path Found
            
            for neighbor in self.neighbors(curr_cell):
                cost_to_neighbor = self.cost_to[curr_cell] + self.weight(curr_cell, neighbor)
                if cost_to_neighbor < self.cost_to[neighbor]: # New path is better than previous best
                    self.predecessors[neighbor] = curr_cell # Update Predecessors
                    self.cost_to[neighbor] = cost_to_neighbor # Update Cost Map
                    self.est_total_cost[neighbor] = cost_to_neighbor + self.heuristic(neighbor, self.goal)

                    self.update_pq(neighbor, cost_to_neighbor)

        return None # No path to the goal

    def recover_path(self):
        path = [self.goal]
        curr_cell = self.goal
        while curr_cell != self.start:
            prev_cell = self.predecessors[curr_cell]
            path.append(prev_cell)
            curr_cell = prev_cell

        return path[::-1] # Reverse path to go from start to goal


    def update_pq(self, cell, new_priority):
        for idx, (old_priority, item) in enumerate(self.pq):
            if item == cell:
                self.pq[idx] = (new_priority, item) # Change only the priority
                heapq._siftdown(self.pq, 0, idx)  # Moved up in priority
                return
            
        # Not already in priority queue
        heapq.heappush(self.pq, (new_priority, cell))

        



