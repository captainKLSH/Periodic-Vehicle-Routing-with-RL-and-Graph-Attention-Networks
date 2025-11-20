import torch
import numpy as np

class VRPDataGenerator:
    def __init__(self, batch_size, num_nodes):
        self.batch_size = batch_size
        self.num_nodes = num_nodes

    def get_data(self):
        # Coords: [Batch, Nodes+1, 2] (0 is Depot)
        coords = torch.rand(self.batch_size, self.num_nodes + 1, 2)
        demands = torch.rand(self.batch_size, self.num_nodes + 1) * 0.4 + 0.1
        demands[:, 0] = 0 # Depot has no demand
        return coords, demands

def calculate_distance(coords, tour):
    """ Calculates total distance including return to depot """
    # tour: [steps, batch, 1] -> [batch, steps]
    tour = tour.permute(1, 0, 2).squeeze(2)
    
    # Gather coordinates
    # coords: [batch, nodes, 2]
    ordered_coords = torch.gather(coords, 1, tour.unsqueeze(2).expand(-1, -1, 2))
    
    # 1. Distance between steps
    step_dists = torch.norm(ordered_coords[:, 1:] - ordered_coords[:, :-1], dim=2)
    
    # 2. Return to depot distance
    # Depot is always at index 0 of 'coords'
    depot_pos = coords[:, 0, :]
    last_pos = ordered_coords[:, -1, :]
    return_dist = torch.norm(depot_pos - last_pos, dim=1)
    
    return step_dists.sum(1) + return_dist