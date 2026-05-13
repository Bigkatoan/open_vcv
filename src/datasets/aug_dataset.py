import os
import random
from pathlib import Path
from typing import Optional, Callable

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image


# ===========================================================================
# COCO Dataset (unsupervised — chỉ cần ảnh)
# ===========================================================================

class CocoImageDataset(Dataset):
    """
    Đọc ảnh từ data/coco2017/train2017/ (hoặc val2017/).
    Không dùng annotations — phù hợp unsupervised learning.
    Trả về (PIL.Image, 0) để tương thích với AugmentedDataset.
    """
    def __init__(self, root: str, split: str = 'train'):
        assert split in ('train', 'val'), "split phải là 'train' hoặc 'val'"
        split_dir = Path(root) / f'{split}2017'
        assert split_dir.exists(), f"Không tìm thấy {split_dir}\nChạy: python scripts/download_coco.py"

        self.imgs = sorted(split_dir.glob('*.jpg'))
        assert len(self.imgs) > 0, f"Không có ảnh trong {split_dir}"
        print(f"[CocoImageDataset] {split}: {len(self.imgs):,} ảnh từ {split_dir}")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert('RGB')
        return img, 0   # label = 0 (dummy, không dùng)


# ===========================================================================
# Augmented Dataset — hỗ trợ cả CIFAR-like và COCO
# ===========================================================================

class AugmentedDataset(Dataset):
    """
    Wraps bất kỳ dataset nào trả về (PIL.Image, label).
    Với mỗi sample, trả về:
        core_imgs: (q, C, H, W) — q augmented versions [0, 1]
        neg_imgs:  (k, C, H, W) — k negative samples   [0, 1]

    q augmentation levels:
        0 — Conservative: flip + small crop
        1 — Moderate:     + color jitter + rotation
        2 — Aggressive:   + grayscale + strong jitter + erasing

    img_size: resize về img_size × img_size trước khi augment
              (quan trọng với COCO vì ảnh có kích thước không đồng nhất)
    """

    def __init__(self, base_dataset, q: int = 3, k: int = 2,
                 img_size: int = 224):
        assert 1 <= q <= 3, "q phải trong [1, 3]"
        self.base     = base_dataset
        self.q        = q
        self.k        = k
        self.img_size = img_size

        pad = max(4, img_size // 16)    # nhỏ hơn với ảnh lớn
        scale = (0.2, 1.0)              # SimCLR standard — phù hợp cho 224px ImageNet

        self._augs = [
            # 0: Conservative
            T.Compose([
                T.Resize(img_size + pad),
                T.RandomHorizontalFlip(0.5),
                T.RandomCrop(img_size),
                T.ToTensor(),
            ]),
            # 1: Moderate
            T.Compose([
                T.Resize(img_size + pad),
                T.RandomHorizontalFlip(0.5),
                T.RandomCrop(img_size),
                T.ColorJitter(0.4, 0.4, 0.4, 0.1),
                T.RandomRotation(15),
                T.ToTensor(),
            ]),
            # 2: Aggressive
            T.Compose([
                T.RandomResizedCrop(img_size, scale=scale),
                T.RandomHorizontalFlip(0.5),
                T.ColorJitter(0.8, 0.8, 0.8, 0.2),
                T.RandomGrayscale(0.2),
                T.RandomRotation(30),
                T.ToTensor(),
                T.RandomErasing(p=0.5, scale=(0.02, 0.2)),
            ]),
        ]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]   # PIL Image

        # q augmented versions
        core_imgs = torch.stack([self._augs[i](img) for i in range(self.q)])

        # k negatives: random samples, conservative aug
        # k=0 → in-batch negatives (được xử lý trong trainer, không cần load thêm)
        if self.k == 0:
            return core_imgs, torch.empty(0)

        neg_indices = random.choices(range(len(self.base)), k=self.k)
        neg_imgs = torch.stack([
            self._augs[0](self.base[ni][0]) for ni in neg_indices
        ])

        return core_imgs, neg_imgs
