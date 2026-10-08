from pathlib import Path
from torchvision.transforms.v2 import functional as F
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data.distributed import DistributedSampler
from common import set_worker_seed


def worm_transforms(img, img_size, upscale=False, augment=False):
    img = F.to_image(img) # uint8 tensor 0-255
    _, h, w = img.shape
    longest = max(h, w)
    
    # do resize aspect ratio and pad for image larger than input size
    if longest > img_size or upscale:
        scale = img_size / longest # value < 1
        h = min(img_size, max(1, round(h * scale)))
        w = min(img_size, max(1, round(w * scale)))
        img = F.resize(img, [h, w], antialias=True)

    pad_h, pad_w = img_size - h, img_size - w # pad range
    padding = [pad_w // 2,              # ขนาดด้านที่จะ pad: ซ้าย บน ขวา ล่าง (F.pad() input)
               pad_h // 2,              # pad_w - (pad_w  // 2) เขียนแบบนี้เพราะพอรวม pixel จะได้ครบ image size เวลา pad size เป็นเลขคี่
               pad_w - (pad_w  // 2),   # เช่น target_w=224: img_w=123 pad_w=224-123=101 left=50, right=101-50=51
               pad_h - (pad_h // 2)]    # 123+50+51 = 224
    img = F.pad(img, padding=padding, fill=0)
    
    if augment:
        if torch.rand(1).item() < 0.5:
            img = F.horizontal_flip(img)
        if torch.rand(1).item() < 0.5:
            img = F.vertical_flip(img)
        k = int(torch.randint(0, 4, (1,)).item()) # สุ่มเลข 0-3
        if k:
            img = torch.rot90(img, k, dims=(-2, -1)) # (N, C, H, W) -> H, W = หมุนตามแกน H, W
            
    img = F.to_dtype(img, dtype=torch.float32, scale=True)
    img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalize
    return img


def get_dataloaders(dataset_dir, train_transform, eval_transform, batch_size, num_workers, pin_memory, is_distributed=False, is_main=True, seed=42):
    ds_dir = Path(dataset_dir)
    train_dir, val_dir, test_dir = ds_dir / 'train', ds_dir / 'val', ds_dir / 'test'

    train_ds = ImageFolder(root=train_dir, transform=train_transform)
    val_ds = ImageFolder(root=val_dir, transform=eval_transform)
    test_ds = ImageFolder(root=test_dir, transform=eval_transform) if test_dir.exists() else None
    
    # check class folder name must be the same for all split
    for name, ds in (('val', val_ds), ('test', test_ds)):
        if ds is not None and ds.class_to_idx != train_ds.class_to_idx:
            raise ValueError(f'Class folders differ between train and {name} '
                             f'train={train_ds.class_to_idx} {name}={ds.class_to_idx}')

    cls_names = train_ds.classes
    num_class = len(cls_names)

    if is_main:
        print(f'Train loaded {len(train_ds)} images')
        print(f'Val loaded {len(val_ds)} images')
        print(f'Test loaded {len(test_ds)} images' if test_ds is not None else 'Test loaded 0 images (no test folder)')
        print(f'Found {num_class} classes: {cls_names}')

    # กระจาย data ให้ gpu แต่ละใบจะได้ข้อมูลไม่ซ้ำกัน
    train_sampler = DistributedSampler(train_ds, shuffle=True, seed=seed) if is_distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_distributed else None
    test_sampler = DistributedSampler(test_ds, shuffle=False) if is_distributed and test_ds is not None else None

    persistent = num_workers > 0 # ไม่ kill cpu process
    prefetch = 2 if persistent else None # เตรียมภาพรอล่วงหน้า n batch
    
    common_kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, worker_init_fn=set_worker_seed, persistent_workers=persistent, prefetch_factor=prefetch)

    # กระจายแล้วเอาเข้า loader
    train_loader = DataLoader(train_ds, shuffle=(train_sampler is None), sampler=train_sampler, drop_last=True, **common_kw) # ห้ามสุ่มซ้ำถ้าทำ ddp เพราะสุ่มใน sampler ไปแล้ว
    val_loader = DataLoader(val_ds, shuffle=False, sampler=val_sampler, **common_kw)
    test_loader = DataLoader(test_ds, shuffle=False, sampler=test_sampler, **common_kw)

    return train_loader, val_loader, test_loader, cls_names, num_class, train_sampler