import torch
import yaml
import os
from models.swin_depth_t import SwinDepthT

with open("configs/swin_simple.yaml", "r") as f:
    config = yaml.safe_load(f)

model = SwinDepthT(config['model'])

# If you have trained weights:
# model.load_state_dict(torch.load("path_to_your_trained_swin_simple.pth"))

os.makedirs("weights", exist_ok=True)
torch.save(model.state_dict(), "weights/swin_simple.pth")
print("✅ swin_simple.pth saved in weights/")
