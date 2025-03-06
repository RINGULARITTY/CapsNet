import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import os
from torchvision import datasets
from torch.utils.data import DataLoader

class AlbumentationsTransform:
    def __init__(self, albumentations_transform):
        self.albumentations_transform = albumentations_transform
        
    def __call__(self, img):
        img = img.convert("L")
        img_np = np.array(img)
        augmented = self.albumentations_transform(image=img_np)
        return augmented["image"]

class CachedDatasetMulti(Dataset):
    def __init__(self, dataset, transform, n_variants=5):
        self.cached_data = []
        self.n_variants = n_variants
        self.classes = getattr(dataset, 'classes', None)
        
        for idx in tqdm(range(len(dataset))):
            img, label = dataset[idx]
            variants = []
            for _ in range(n_variants):
                transformed = transform(img)
                variants.append(transformed)
            self.cached_data.append((variants, label))

    def __getitem__(self, index):
        variants, label = self.cached_data[index]
        return random.choice(variants), label

    def __len__(self):
        return len(self.cached_data)

def load_raw_datasets(dataset_name):
    train_dataset_raw = datasets.ImageFolder(root=f"datasets/{dataset_name}/train", transform=None)
    val_dataset_raw   = datasets.ImageFolder(root=f"datasets/{dataset_name}/valid", transform=None)
    test_dataset_raw  = datasets.ImageFolder(root=f"datasets/{dataset_name}/test", transform=None)
    
    return train_dataset_raw, val_dataset_raw, test_dataset_raw
    

def load_trainset_with_resize(img_size, train_dataset_raw, val_dataset_raw):
    albumentations_train_transform = A.Compose([
        A.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.9, 1.0),         # scale variations
            ratio=(0.95, 1.05),       # limit deformation ratio
            interpolation=1,          # cv2.INTER_LINEAR
            mask_interpolation=0,     # cv2.INTER_NEAREST
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),      # slightly change orientation
        A.Rotate(limit=10, p=0.5),    # slightly change rotation
        A.ShiftScaleRotate(
            shift_limit=0.05,         # slight shifts
            scale_limit=0.1,          # slight scale change
            rotate_limit=0,
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.RandomGamma(
            gamma_limit=(80, 120),    # slightly change gamma
            p=0.5
        ),
        A.Blur(blur_limit=3, p=0.3),  # slightly change blur
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])
    
    albumentations_val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])

    train_transform = AlbumentationsTransform(albumentations_train_transform)
    val_transform = AlbumentationsTransform(albumentations_val_transform)
    
    train_dataset = CachedDatasetMulti(train_dataset_raw, train_transform, n_variants=10)
    val_dataset = CachedDatasetMulti(val_dataset_raw, val_transform, n_variants=1)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True, pin_memory=True)
    
    return train_loader, val_loader

def load_test_dataset(dataset_name, img_size):
    test_dataset_raw  = datasets.ImageFolder(root=f"datasets/{dataset_name}/test", transform=None)

    albumentations_test_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])
    
    test_transform = AlbumentationsTransform(albumentations_test_transform)

    test_dataset = CachedDatasetMulti(test_dataset_raw, test_transform, n_variants=1)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, pin_memory=True)
    
    return test_dataset, test_loader
    

class CacheLoader:
    def __init__(self, train_dataset_raw, val_dataset_raw):
        self.cache = {}
        self.train_dataset_raw = train_dataset_raw
        self.val_dataset_raw = val_dataset_raw
    
    def get_or_create_cache(self, img_size):
        if img_size in self.cache:
           return self.cache[img_size]
       
        self.cache[img_size] = load_trainset_with_resize(img_size, self.train_dataset_raw, self.val_dataset_raw)

        return self.cache[img_size]