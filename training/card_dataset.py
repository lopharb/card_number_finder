import torch
import os
import cv2
from torch.utils.data import Dataset


class CardSegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))]
        self.transform = transform or (lambda x: {"image": x, "bboxes": [], "class_labels": []})

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, os.path.splitext(self.image_files[idx])[0] + ".txt")

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks = []
        class_labels = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    cls = int(parts[0])
                    coords = list(map(float, parts[1:]))
                    if len(coords) % 2 == 0:
                        masks.append(coords)
                        class_labels.append(cls)

        # You can apply augmentations here (currently dummy)
        return {
            'image': torch.tensor(image).permute(2, 0, 1).float() / 255.0,
            'masks': torch.tensor(masks, dtype=torch.float32),
            'labels': torch.tensor(class_labels)
        }
