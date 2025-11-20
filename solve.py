import torch
import numpy as np
from src.model import AttentionDynamicModel
from visualize import visualize_routes 

def load_trained_model(model_path, device):
    """
    Reconstructs the AI's 'Brain' and loads the memories (weights).
    """
    # 1. Rebuild the architecture (must match training exactly)
    model = AttentionDynamicModel(embedding_dim=128, n_heads=8, n_layers=3)
    
    # 2. Load weights
    # map_location ensures it loads on CPU even if trained on GPU
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    # 3. Evaluation Mode (Stop learning, start working)
    model.eval() 
    model.to(device)
    
    return model

def solve_problem(model, data, device):
    """
    Asks the AI to solve a specific map.
    """
    # 1. Format the Map (Coordinates)
    # Shape: [Batch_Size, Nodes, 2]
    inputs = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)

    # 2. Create Dummy Demands (The Fix)
    # The model needs to know package weights. Since this is a simple visual demo,
    # we assign a small weight (0.1) to every customer so the truck never gets "full".
    batch_size, num_nodes, _ = inputs.size()
    demands = torch.zeros(batch_size, num_nodes, device=device)
    demands[:, 1:] = 0.1  # Customers have 0.1 demand, Depot (index 0) has 0
    
    # 3. Run the model
    # decode_type="greedy" means: "Always pick the best option."
    with torch.no_grad():
        model.decode_type = "greedy"
        # We now pass 'demands' to the model!
        _, _, tour = model(inputs, demands, return_pi=True)

    # 4. Calculate actual distance for the display
    # We do this manually here to get the specific number for the title
    # (The model returns cost, but extracting it from the complex return tuple is cleaner here)
    tour_indices = tour.cpu().numpy()[0]
    return tour_indices

if __name__ == "__main__":
    # --- 1. Configuration ---
    MODEL_FILE = "vrp_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 2. The Map (Input Data) ---
    # Depot is first (0.5, 0.5), followed by 5 customers
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
        route_indices = solve_problem(brain, customers, device)
        
        print(f"\nOptimized Route Sequence: {route_indices}")
        
        # --- 5. Visualize ---
        print("Generating Map...")
        depot = customers[0]
        # Note: visualize_routes expects the customer list to NOT include depot,
        # but the route_indices refer to the full list (0=Depot).
        # Our visualize.py handles this by mapping indices to the combined list.
        visualize_routes(depot, customers[1:], route_indices, title="AI Optimized Solution")

    except FileNotFoundError:
        print("\n❌ Error: 'vrp_model.pth' not found.")
        print("   Please run 'python main.py' first to train the model!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")