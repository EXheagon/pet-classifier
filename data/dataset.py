import torch
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# =========================
# 1. 下载/加载 Oxford-IIIT Pet
# =========================

trainval_data = datasets.OxfordIIITPet(
    root="./data",
    split="trainval",
    target_types="category",
    download=True
)

test_data = datasets.OxfordIIITPet(
    root="./data",
    split="test",
    target_types="category",
    download=False
)


# =========================
# 2. 生成 70% / 15% / 15% 划分
# =========================

all_labels = np.concatenate([
    np.array(trainval_data._labels),
    np.array(test_data._labels)
])

all_indices = np.arange(len(all_labels))

train_idx, temp_idx = train_test_split(
    all_indices,
    test_size=0.30,
    stratify=all_labels,
    random_state=42
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    stratify=all_labels[temp_idx],
    random_state=42
)


# =========================
# 3. 数据增强
# =========================

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


train_transform_improved = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# 4. 自定义 Dataset
# =========================

class PetDataset(Dataset):

    def __init__(self, base_dataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        image, label = self.base_dataset[real_idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# =========================
# 5. 合并数据
# =========================

all_data = torch.utils.data.ConcatDataset([
    trainval_data,
    test_data
])


# =========================
# 6. 创建 Dataset
# =========================

train_dataset = PetDataset(
    all_data,
    train_idx,
    train_transform
)

train_dataset_improved = PetDataset(
    all_data,
    train_idx,
    train_transform_improved
)

val_dataset = PetDataset(
    all_data,
    val_idx,
    eval_transform
)

test_dataset = PetDataset(
    all_data,
    test_idx,
    eval_transform
)


# =========================
# 7. 创建 DataLoader
# =========================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2
)

train_loader_improved = DataLoader(
    train_dataset_improved,
    batch_size=32,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2
)
