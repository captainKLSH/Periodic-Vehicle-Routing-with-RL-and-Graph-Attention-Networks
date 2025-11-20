import torch

class VRPAgent:
    def __init__(self, input_data, capacity=1.0):
        self.batch_size, self.n_nodes, _ = input_data.size()
        self.coords = input_data
        self.capacity = capacity
        self.device = input_data.device
        
        self.prev_node = torch.zeros(self.batch_size, 1, dtype=torch.long, device=self.device)
        self.used_capacity = torch.zeros(self.batch_size, 1, device=self.device)
        self.visited = torch.zeros(self.batch_size, 1, self.n_nodes, dtype=torch.uint8, device=self.device)
        self.step_count = 0

    def all_finished(self):
        return self.visited[:, :, 1:].all()

    def get_mask(self, demands):
        mask = self.visited.clone() > 0
        
        remaining_capacity = self.capacity - self.used_capacity
        cant_fit = (demands > remaining_capacity).unsqueeze(1)
        mask = mask | cant_fit
        
        # Allow return to depot (index 0) if we moved at least once
        if self.step_count > 0:
            mask[:, :, 0] = False
            
        # If stuck (all true), unmask depot to prevent crash
        is_stuck = mask.all(dim=2)
        mask[is_stuck.squeeze(-1), 0, 0] = False
        
        return mask

    def step(self, next_node, demands):
        self.step_count += 1
        batch_indices = torch.arange(self.batch_size, device=self.device)
        
        # Mark visited
        self.visited[batch_indices, 0, next_node.squeeze()] = 1
        
        # Update Capacity
        is_depot = (next_node == 0).float()
        current_demand = torch.gather(demands, 1, next_node)
        self.used_capacity = (self.used_capacity + current_demand) * (1 - is_depot)
        self.prev_node = next_node