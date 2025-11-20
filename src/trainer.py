import torch
import torch.optim as optim
from src.data import calculate_distance

class BaselineTrainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        
        # Simple Moving Average Baseline
        self.baseline = None
        self.decay = 0.9

    def train_step(self, coords, demands):
        self.model.train()
        self.model.decode_type = "sampling"
        
        # 1. Forward Pass
        log_probs_list, tour_indices, _ = self.model(coords, demands)
        
        # 2. Calculate Cost (Total Distance)
        # tour_indices: [steps, batch, 1]
        cost = calculate_distance(coords, tour_indices)
        
        # 3. Reinforcement Learning (REINFORCE)
        
        # Update Baseline
        if self.baseline is None:
            self.baseline = cost.mean().item()
        else:
            self.baseline = self.decay * self.baseline + (1 - self.decay) * cost.mean().item()
            
        # --- CRITICAL FIX START ---
        # Advantage: How much BETTER is this run?
        # If Cost (10) < Baseline (15), then Advantage should be +5 (Positive/Good)
        advantage = (self.baseline - cost)
        # --- CRITICAL FIX END ---
        
        # 4. Calculate Loss
        # Sum log probs over the steps to get log_prob of the whole tour
        tour_log_probs = []
        for i, step_log_prob in enumerate(log_probs_list):
            selected = tour_indices[i] 
            prob = step_log_prob.gather(1, selected)
            tour_log_probs.append(prob)
            
        tour_log_probs = torch.cat(tour_log_probs, dim=1).sum(1) # [batch]
        
        # Loss = - (Advantage * Log_Prob)
        # We want to maximize Advantage, so we minimize Negative Advantage
        loss = -(advantage * tour_log_probs).mean()
        
        # 5. Backprop
        self.optimizer.zero_grad()
        loss.backward()
        
        # --- SAFETY FIX: Gradient Clipping ---
        # Prevents the "Brain" from changing too drastically at once
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return cost.mean().item(), loss.item()