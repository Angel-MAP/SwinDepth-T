import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class DepthDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.depth_dir = os.path.join(root_dir, 'depths')
        self.image_files = sorted(os.listdir(self.image_dir))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        depth_path = os.path.join(self.depth_dir, self.image_files[idx].replace('.jpg', '.png'))

        image = Image.open(image_path).convert('RGB')
        depth = Image.open(depth_path)

        return self.transform(image), self.transform(depth)
