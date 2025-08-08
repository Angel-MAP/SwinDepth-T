import torch
import yaml
from models import swin_depth_t
from utils import dataloader, evaluation
from torch.utils.data import DataLoader

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def evaluate(config):
    # Load dataset
    test_set = dataloader.DepthDataset(config['dataset']['test'])
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # Load model
    model = swin_depth_t.SwinDepthT(config['model'])
    model.load_state_dict(torch.load(config['evaluation']['weights_path'], map_location=config['device']))
    model = model.to(config['device'])
    model.eval()

    # Evaluation loop
    with torch.no_grad():
        total_metrics = {'abs_rel': 0, 'sq_rel': 0, 'rmse': 0, 'rmse_log': 0, 'a1': 0, 'a2': 0, 'a3': 0}
        for images, depths in test_loader:
            images, depths = images.to(config['device']), depths.to(config['device'])
            preds = model(images)
            metrics = evaluation.compute_metrics(preds, depths)
            for key in total_metrics:
                total_metrics[key] += metrics[key]

    # Average results
    num_samples = len(test_loader)
    for key in total_metrics:
        total_metrics[key] /= num_samples

    print("Evaluation Results:")
    for k, v in total_metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    config_path = "configs/swin_tiny.yaml"
    config = load_config(config_path)
    evaluate(config)
