import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, input_dim, embed_dim=None):
        super(MultiHeadAttention, self).__init__()
        
        if embed_dim is None: embed_dim = input_dim
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // n_heads
        
        assert self.head_dim * n_heads == embed_dim, "Embed dim must be divisible by n_heads"
        
        # Standard Linear Layers for Q, K, V
        self.w_q = nn.Linear(input_dim, embed_dim)
        self.w_k = nn.Linear(input_dim, embed_dim)
        self.w_v = nn.Linear(input_dim, embed_dim)
        self.w_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k=None, v=None, mask=None):
        if k is None: k = q
        if v is None: v = q
        
        batch_size = q.size(0)
        
        # 1. Linear Projections
        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)
        
        # 2. Split Heads
        # Transform: [batch, nodes, embed] -> [batch, nodes, heads, head_dim] -> [batch, heads, nodes, head_dim]
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 3. Attention Score
        # Shape: [batch, heads, q_nodes, k_nodes]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            # --- DEBUG FIX START ---
            # Previous code: mask.unsqueeze(1) -> Shape [Batch, 1, Nodes]
            # Required Broadcast: [Batch, Heads, Q_Nodes, K_Nodes]
            # Correct Shape: [Batch, 1, 1, Nodes] to broadcast over Heads and Query Nodes
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) > 0, -1e9)
            # --- DEBUG FIX END ---
            
        weights = F.softmax(scores, dim=-1)
        
        # 4. Combine Heads
        out = torch.matmul(weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        return self.w_out(out)

class GraphAttentionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads, n_layers):
        super(GraphAttentionEncoder, self).__init__()
        self.init_embed = nn.Linear(input_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            MultiHeadAttention(n_heads, hidden_dim) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Linear(hidden_dim * 4, hidden_dim)
            ) for _ in range(n_layers)
        ])
        self.ff_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])

    def forward(self, x):
        h = self.init_embed(x)
        for i in range(len(self.layers)):
            # Attention Block (Post-Norm)
            h = self.norms[i](h + self.layers[i](h))
            # Feed Forward Block (Post-Norm)
            h = self.ff_norms[i](h + self.ff_layers[i](h))
            
        # Return Node Embeddings and Graph Embedding (Mean Pooling)
        return h, torch.mean(h, dim=1)