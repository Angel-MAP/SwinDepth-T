import torch
import yaml
from models import swin_depth_t
from utils import dataloader, evaluation
from torch.utils.data import DataLoader

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def train(config):
    # Load dataset
    train_set = dataloader.DepthDataset(config['dataset']['train'])
    train_loader = DataLoader(train_set, batch_size=config['training']['batch_size'], shuffle=True)

    # Load model
    model = swin_depth_t.SwinDepthT(config['model'])
    model = model.to(config['device'])

    # Loss, optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])

    # Training loop
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        for images, depths in train_loader:
            images, depths = images.to(config['device']), depths.to(config['device'])
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, depths)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    # Save model
    torch.save(model.state_dict(), config['training']['save_path'])
    print("Training complete. Model saved.")

if __name__ == "__main__":
    config_path = "configs/swin_tiny.yaml"
    config = load_config(config_path)
    train(config)
