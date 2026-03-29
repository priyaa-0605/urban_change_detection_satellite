import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class LEVIRDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.before_dir = os.path.join(root_dir, "A")
        self.after_dir = os.path.join(root_dir, "B")
        self.label_dir = os.path.join(root_dir, "label")

        # Get all filenames from A folder
        self.file_names = sorted(os.listdir(self.before_dir))

        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]

        before_path = os.path.join(self.before_dir, file_name)
        after_path = os.path.join(self.after_dir, file_name)
        label_path = os.path.join(self.label_dir, file_name)

        before = Image.open(before_path).convert("RGB")
        after = Image.open(after_path).convert("RGB")
        label = Image.open(label_path).convert("L")

        before = self.transform(before)
        after = self.transform(after)

        label = self.transform(label)
        label = (label > 0).float()  # Convert to 0 or 1

        return before, after, label