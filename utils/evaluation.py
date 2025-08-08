import torch
import torch.nn.functional as F

def compute_metrics(pred, gt):
    """
    pred: predicted depth map (batch_size x 1 x H x W)
    gt: ground truth depth map (batch_size x 1 x H x W)
    Returns dict with error metrics.
    """
    pred = pred.squeeze().detach().cpu()
    gt = gt.squeeze().detach().cpu()

    pred = torch.clamp(pred, min=1e-3, max=80)
    gt = torch.clamp(gt, min=1e-3, max=80)

    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean().item()
    a2 = (thresh < 1.25 ** 2).float().mean().item()
    a3 = (thresh < 1.25 ** 3).float().mean().item()

    abs_rel = torch.mean(torch.abs(gt - pred) / gt).item()
    sq_rel = torch.mean(((gt - pred) ** 2) / gt).item()
    rmse = torch.sqrt(torch.mean((gt - pred) ** 2)).item()
    rmse_log = torch.sqrt(torch.mean((torch.log(gt) - torch.log(pred)) ** 2)).item()

    return {
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'rmse': rmse,
        'rmse_log': rmse_log,
        'a1': a1,
        'a2': a2,
        'a3': a3
    }
