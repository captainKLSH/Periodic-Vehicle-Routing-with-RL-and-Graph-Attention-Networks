import torch
import numpy as np
import json
from src.model import AttentionDynamicModel
from visualize import visualize_routes  # Using your visualizer

def load_trained_model(model_path, device):
    """
    Reconstructs the AI's 'Brain' and loads the memories (weights) 
    from the saved file.
    """
    # 1. Rebuild the architecture (must match the training settings exactly)
    model = AttentionDynamicModel(embedding_dim=128, n_heads=8, n_layers=3)
    
    # 2. Load the saved weights
    # map_location ensures it works even if you trained on GPU but run on CPU
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    # 3. Set to "Evaluation Mode"
    # This tells the model: "Stop learning, start working."
    model.eval() 
    model.to(device)
    
    return model

def solve_problem(model, data, device):
    """
    Asks the AI to solve a specific map.
    """
    # Prepare the data
    # The model expects a "batch", so we add a dimension (unsqueeze)
    # Format: [Batch_Size, Nodes, Coordinates]
    inputs = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)

    # Run the model
    # decode_type="greedy" means: "Always pick the absolute best option you see."
    # (As opposed to experimenting with random options like during training)
    with torch.no_grad(): # Don't calculate gradients (saves memory)
        model.decode_type = "greedy"
        cost, log_likelihood, tour = model(inputs, return_pi=True)

    # Clean up output
    # .cpu().numpy() moves the data from the video card back to normal memory
    return tour.cpu().numpy()[0], cost.item()

if __name__ == "__main__":
    # --- 1. Configuration ---
    MODEL_FILE = "vrp_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 2. The Map (Input Data) ---
    # For this demo, we manually define a Depot (0,0) and 5 customers
    # In the real world, you would load this from a JSON or CSV file
    # Format: [x, y]
    customers = [
        [0.50, 0.50], # Depot (Index 0)
        [0.12, 0.34], # Customer 1
        [0.89, 0.21], # Customer 2
        [0.45, 0.88], # Customer 3
        [0.77, 0.10], # Customer 4
        [0.33, 0.65]  # Customer 5
    ]
    
    print("--- Loading AI Navigator ---")
    
    try:
        # --- 3. Load the Brain ---
        brain = load_trained_model(MODEL_FILE, device)
        print("Model loaded successfully.")

        # --- 4. Solve the Problem ---
        print(f"Solving for {len(customers)-1} customers...")
        route_indices, total_distance = solve_problem(brain, customers, device)
        
        print(f"\nOptimized Route Sequence: {route_indices}")
        print(f"Total Estimated Distance: {total_distance:.4f}")
        
        # --- 5. Visualize ---
        print("Generating Map...")
        depot = customers[0]
        # Exclude depot from the 'nodes' list passed to visualize because 
        # visualize_routes likely handles the depot separately or as part of the list
        # We pass the full list and let the visualizer handle the indices
        visualize_routes(depot, customers[1:], route_indices, title=f"Solution (Dist: {total_distance:.2f})")

    except FileNotFoundError:
        print("Error: Could not find 'vrp_model.pth'. Please run 'main.py' first to train the model!")