import torch
import torch.optim as optim
from src.model import AttentionDynamicModel
from src.trainer import BaselineTrainer
from src.data import VRPDataGenerator
import os

def main():
    # Configuration
    EPOCHS = 200
    BATCH_SIZE = 32
    NUM_NODES = 20
    LR = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Starting VRP Training on {DEVICE}...")
    
    # Initialize
    model = AttentionDynamicModel(embedding_dim=128, n_heads=8, n_layers=3).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    trainer = BaselineTrainer(model, optimizer, DEVICE)
    generator = VRPDataGenerator(BATCH_SIZE, NUM_NODES)
    
    # Training Loop
    for epoch in range(EPOCHS):
        # Get new random map
        coords, demands = generator.get_data()
        coords = coords.to(DEVICE)
        demands = demands.to(DEVICE)
        
        # Train Step
        avg_dist, loss = trainer.train_step(coords, demands)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Distance = {avg_dist:.4f} | Loss = {loss:.4f}")
            
    # Save
    torch.save(model.state_dict(), "vrp_model.pth")
    print("Training Complete. Model saved as vrp_model.pth")

if __name__ == "__main__":
    main()