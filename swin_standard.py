import torch
import yaml
import os
from models.swin_depth_t import SwinDepthT

with open("configs/swin_standard.yaml", "r") as f:
    config = yaml.safe_load(f)

model = SwinDepthT(config['model'])

# If you have trained weights:
# model.load_state_dict(torch.load("path_to_your_trained_swin_standard.pth"))

os.makedirs("weights", exist_ok=True)
torch.save(model.state_dict(), "weights/swin_standard.pth")
print("✅ swin_standard.pth saved in weights/")
