from pathlib import Path
from torchvision.transforms.v2 import functional as F
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data.distributed import DistributedSampler
from common import set_worker_seed

def worm_transforms(img, img_size):
    img = F.to_image(img)

    _, h, w = img.shape
    scale = img_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img = F.resize(img, [new_h, new_w], antialias=True)

    pad_h = img_size - new_h
    pad_w = img_size - new_w

    padding = [
        pad_w // 2,
        pad_h // 2,
        pad_w - (pad_w // 2),
        pad_h - (pad_h // 2)
    ]

    img = F.pad(img, padding=padding, fill=0)
    img = F.to_dtype(img, dtype=torch.float32, scale=True)
    img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return img

def get_dataloaders(dataset_dir, transform, batch_size, num_workers, pin_memory, is_distributed=False, is_main=True):
    ds_dir = Path(dataset_dir)
    train_dir, val_dir, test_dir = ds_dir / 'train', ds_dir / 'val', ds_dir / 'test'

    train_ds = ImageFolder(root=train_dir, transform=transform)
    val_ds = ImageFolder(root=val_dir, transform=transform)
    test_ds = ImageFolder(root=test_dir, transform=transform) if test_dir.exists() else None

    class_names = train_ds.classes
    num_classes = len(class_names)

    if is_main:
        print(f'Train loaded {len(train_ds)} images')
        print(f'Val loaded {len(val_ds)} images')
        if test_ds is not None:
            print(f'Test loaded {len(test_ds)} images')
        else:
            print('Test loaded 0 images (no test folder)')
        print(f'Found {num_classes} classes: {class_names}')

    train_sampler = DistributedSampler(train_ds) if is_distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_distributed else None
    test_sampler = DistributedSampler(test_ds, shuffle=False) if is_distributed and test_ds is not None else None

    persistent = num_workers > 0
    prefetch = 2 if persistent else None

    loaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=(train_sampler is None),
                            sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory,
                            worker_init_fn=set_worker_seed,
                            persistent_workers=persistent, prefetch_factor=prefetch),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          sampler=val_sampler, num_workers=num_workers, pin_memory=pin_memory,
                          worker_init_fn=set_worker_seed,
                          persistent_workers=persistent, prefetch_factor=prefetch),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                           sampler=test_sampler, num_workers=num_workers, pin_memory=pin_memory,
                           worker_init_fn=set_worker_seed,
                           persistent_workers=persistent, prefetch_factor=prefetch) if test_ds is not None else None
    }

    return loaders['train'], loaders['val'], loaders['test'], class_names, num_classes, train_sampler