import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers import GraphAttentionEncoder
from src.env import VRPAgent

class AttentionDynamicModel(nn.Module):
    def __init__(self, embedding_dim=128, n_heads=8, n_layers=3):
        super(AttentionDynamicModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.encoder = GraphAttentionEncoder(input_dim=2, hidden_dim=embedding_dim, n_heads=n_heads, n_layers=n_layers)
        
        # Decoder projections
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(embedding_dim * 2 + 1, embedding_dim, bias=False)
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        self.decode_type = "sampling" # options: "sampling" or "greedy"

    def forward(self, inputs, demands, return_pi=False):
        """
        inputs: [Batch, Nodes, 2] (Coordinates)
        demands: [Batch, Nodes] (Normalized demands)
        """
        batch_size, n_nodes, _ = inputs.size()
        
        # 1. Encode the graph
        embeddings, mean_graph = self.encoder(inputs)
        
        # 2. Initialize the Environment (The Referee)
        agent = VRPAgent(inputs)
        
        # 3. Prepare Context
        fixed_context = self.project_fixed_context(mean_graph) 
        outputs = []
        pi = []
        
        # 4. Decoding Loop
        # We loop up to n_nodes * 2 to be safe, but usually finish in n_nodes + (returns to depot)
        for _ in range(n_nodes * 2):
            if agent.all_finished(): 
                break
                
            # Construct the "Step Context" (Where am I? How much fuel left?)
            # Gather embedding of the previous node visited
            prev_embed = embeddings.gather(1, agent.prev_node.unsqueeze(2).expand(-1, -1, self.embedding_dim)).squeeze(1)
            remaining_cap = (agent.capacity - agent.used_capacity)
            
            step_context = torch.cat([mean_graph, prev_embed, remaining_cap], dim=1)
            h_c = self.project_step_context(step_context)
            
            # Calculate Query Vector
            query = fixed_context + h_c
            
            # Calculate Attention (Compatibility scores)
            # Shape: [Batch, Nodes]
            compatibility = torch.matmul(embeddings, query.unsqueeze(2)).squeeze(2)
            
            # Apply Mask (Block invalid moves)
            # agent.get_mask returns [Batch, 1, Nodes], we squeeze to [Batch, Nodes]
            mask = agent.get_mask(demands).squeeze(1)
            compatibility.masked_fill_(mask, -1e9)
            
            # Convert to Probabilities
            log_p = F.log_softmax(compatibility, dim=-1)
            
            # Select Next Node
            if self.decode_type == "greedy" and not self.training:
                # Pick best
                selected = torch.argmax(log_p, dim=1).unsqueeze(1)
            else:
                # Sample (for training exploration)
                probs = torch.exp(log_p)
                
                # Safety check for NaN (rare numerical stability issue)
                if torch.isnan(probs).any(): 
                    probs = torch.ones_like(probs) / n_nodes 
                    
                selected = torch.multinomial(probs, 1)
            
            # Store results
            outputs.append(log_p)
            pi.append(selected)
            
            # Move the truck
            agent.step(selected, demands)

        # Return stacked tensors
        # outputs: [Steps, Batch, Nodes]
        # pi: [Steps, Batch, 1]
        return torch.stack(outputs), torch.stack(pi), embeddings