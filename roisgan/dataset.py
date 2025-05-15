import os
import numpy as np
import cv2
from PIL import Image
import albumentations as A
import torch
from torch.utils.data import Dataset

# Dataset
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=(256, 256), split='train'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.split = split
        self.image_paths, self.mask_paths = self.get_file_paths()
        self.mean, self.std = self.compute_statistics()

    def get_file_paths(self):
        image_paths, mask_paths = [], []
        count = 0
        for img_name in os.listdir(self.image_dir):
            img_path = os.path.join(self.image_dir, img_name)
            mask_name1 = img_name.replace('.tif', '_ch02_mask.png')
            mask_name2 = img_name.replace('.tif', 'Mask.tif')
            mask_name3 = img_name.replace('.tif','_mask.png')
            mask_path1 = os.path.join(self.mask_dir, mask_name1)
            mask_path2 = os.path.join(self.mask_dir, mask_name2)
            mask_path3 = os.path.join(self.mask_dir, mask_name3)
            if os.path.exists(mask_path1):
                mask_path = mask_path1
            elif os.path.exists(mask_path2):
                mask_path = mask_path2
            elif os.path.exists(mask_path3):
                mask_path = mask_path3
            else:
                count += 1
                continue
            image_paths.append(img_path)
            mask_paths.append(mask_path)
        print(f"Count of mismatched images and masks: {count}")
        return image_paths, mask_paths

    def compute_statistics(self):
        images = []
        for img_path in self.image_paths[:]:
            img = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32) / 255.0
            img = cv2.resize(img, self.img_size)
            images.append(img)
        images = np.stack(images)
        return images.mean(axis=(0, 1, 2)), images.std(axis=(0, 1, 2))

    def preprocess(self, image, mask):
        image = (image / 255.0 - self.mean) / self.std
        image = cv2.resize(image, self.img_size)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST) / 255.0
        mask = np.expand_dims(mask, axis=-1)
        return image.astype(np.float32), mask.astype(np.float32)

    def augment_image(self, image, mask):
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.3),
            A.RandomBrightnessContrast(p=0.2),
            A.Resize(height=self.img_size[0], width=self.img_size[1], always_apply=True)
        ])
        augmented = transform(image=image, mask=mask)
        return augmented['image'], augmented['mask']

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')
        image, mask = self.preprocess(np.array(img), np.array(mask))
        if self.split == 'train':
            image, mask = self.augment_image(image, mask)
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).permute(2, 0, 1)
        return image, mask