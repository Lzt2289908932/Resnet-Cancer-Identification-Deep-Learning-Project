import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

def get_train_transforms(img_size=224):
    """训练用的augmentation"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2, 0.2, 0.1, 0.05),
        transforms.RandomAffine(0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_test_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class CancerCellDataset(Dataset):

    #读取 data_dir/0/ 和 data_dir/1/ 0=正常 1=癌细胞

    EXTS = {".jpg", ".jpeg", ".png"}

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        for label in [0, 1]:
            d = os.path.join(data_dir, str(label))
            if not os.path.isdir(d):
                print(f"[warn] 目录不存在: {d}")
                continue
            for f in sorted(os.listdir(d)):
                if os.path.splitext(f)[1].lower() in self.EXTS:
                    self.samples.append((os.path.join(d, f), label))

        n0 = sum(1 for _, l in self.samples if l == 0)
        n1 = len(self.samples) - n0
        print(f"loaded {len(self.samples)} images (normal={n0}, cancer={n1})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class UnlabeledDataset(Dataset):
    #无标签的测试集，直接扫文件夹

    EXTS = {".jpg", ".jpeg", ".png"}

    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        for root, _, files in os.walk(data_dir):
            for f in sorted(files):
                if os.path.splitext(f)[1].lower() in self.EXTS:
                    self.image_paths.append(os.path.join(root, f))
        print(f"found {len(self.image_paths)} unlabeled images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, path


class SubsetWithTransform(Dataset):
    """
    random_split出来的subset没法单独设transform，
    所以包一层，训练集和验证集用不同的transform
    """

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # 直接去原始dataset拿路径和label，不走原来的transform
        real_idx = self.subset.indices[idx]
        path, label = self.subset.dataset.samples[real_idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def create_dataloaders_with_split(train_dir, val_ratio=0.2, batch_size=32,
                                   img_size=224, num_workers=4, seed=42):
    #从训练集拆出一部分当验证集（因为测试集没label）

    full_ds = CancerCellDataset(train_dir, transform=None)

    n = len(full_ds)
    n_val = int(n * val_ratio)
    n_train = n - n_val

    g = torch.Generator().manual_seed(seed)
    train_sub, val_sub = random_split(full_ds, [n_train, n_val], generator=g)
    print(f"split: train={n_train}, val={n_val}")

    train_data = SubsetWithTransform(train_sub, get_train_transforms(img_size))
    val_data = SubsetWithTransform(val_sub, get_test_transforms(img_size))

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader
