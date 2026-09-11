import sys

import numpy as np
import torch
from torchvision.io import decode_image, ImageReadMode
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F

sys.path.insert(0, "/home/kklamcha/worm-project/")
from worm_utils import get_full_dataframe

sys.path.insert(0, "/project/lt200264-saiwat/Code/Segmentation")
from mask_boundary import boundary_points_to_mask

from tools import utils

from common import set_worker_seed


def new_class_map(classes):
    return {cls: i + 1 for i, cls in enumerate(classes)}


def get_transforms(train):
    transforms = [T.ToImage()]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.5))
        transforms.append(T.SanitizeBoundingBoxes())  # filter, cleaning bboxes
    transforms.append(T.ToDtype(torch.float32, scale=True))
    return T.Compose(transforms)


class WormDataset(Dataset):
    def __init__(self, df, class_map, transforms=None, name="default"):
        needed_cols = ["stem", "raw_img_path", "Class", "boundary", "bbox"]
        self.df = df[needed_cols].reset_index(drop=True)
        self.img_stems = self.df["stem"].sort_values().unique().tolist()  # ชื่อรูปทั้งหมด

        self._groups = self.df.groupby("stem").indices  # ข้อมูลที่เกี่ยวข้องกับรูป

        self.class_map = class_map
        self.transforms = transforms
        self.name = name

    def __len__(self):
        return len(self.img_stems)

    def __getitem__(self, idx):
        stem = self.img_stems[idx]
        rows = self._groups[stem]
        instances = self.df.iloc[rows]

        img_path = instances["raw_img_path"].iat[0]

        # read image as tensor
        img = decode_image(img_path, mode=ImageReadMode.RGB)
        _, h, w = img.shape  # (C, H, W)

        masks_list, bboxes_list, labels_list = [], [], []
        for cls, boundary, bbox in zip(
            instances["Class"].astype(int), instances["boundary"], instances["bbox"]
        ):
            # convert contour points to mask
            mask = boundary_points_to_mask(boundary, (h, w))
            if mask.sum() == 0:  # ถ้าไม่มี mask
                continue

            # filter bbox
            x, y, bw, bh = bbox
            if x < 0 or y < 0 or bw <= 0 or bh <= 0 or (x + bw) > w or (y + bh) > h:
                print(
                    f"[WormDataset] skipping out-of-bounds bbox in stem {stem!r}: "
                    f"bbox={bbox}, image size=({w}, {h})",
                    flush=True,  # ไม่รอ print เลย
                )
                continue

            masks_list.append(torch.from_numpy(np.ascontiguousarray(mask, dtype=np.uint8)))
            bboxes_list.append(bbox)
            labels_list.append(self.class_map[cls])

        if len(masks_list) == 0:
            masks = torch.empty((0, h, w), dtype=torch.uint8)
            bboxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
            area = torch.empty((0,), dtype=torch.float32)
            iscrowd = torch.empty((0,), dtype=torch.int64)
        else:
            masks = torch.stack(masks_list)  # เอามารวมกันเป็น tensor หลายมิติ (N, H, W) โดย N คือจำนวน mask
            bboxes = torch.as_tensor(np.stack(bboxes_list), dtype=torch.float32)
            labels = torch.as_tensor(labels_list, dtype=torch.int64)
            area = bboxes[:, 2] * bboxes[:, 3]
            iscrowd = torch.zeros((len(masks_list),), dtype=torch.int64)

        del masks_list, bboxes_list, labels_list

        # warper บอก torch ว่าเป็น object อะไร
        bboxes = tv_tensors.BoundingBoxes(bboxes, format="XYWH", canvas_size=(h, w))
        masks = tv_tensors.Mask(masks)
        img = tv_tensors.Image(img)
        img = F.to_dtype(img, dtype=torch.float32, scale=True)

        target = {
            "boxes": bboxes,
            "masks": masks,
            "labels": labels,
            "image_id": torch.tensor(idx),
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        target["boxes"] = F.convert_bounding_box_format(
            target["boxes"], new_format=tv_tensors.BoundingBoxFormat.XYXY
        )

        t_masks = target["masks"]
        t_boxes = target["boxes"]

        # filter empty masks and invalid boxes
        if t_masks.numel() > 0:
            valid_mask = t_masks.sum(dim=(1, 2)) > 0

            wh = t_boxes[:, 2:] - t_boxes[:, :2]
            valid_box = (wh > 0).all(dim=1)

            keep = valid_mask & valid_box

            if not keep.all():
                target["masks"] = tv_tensors.Mask(t_masks[keep])
                target["boxes"] = tv_tensors.BoundingBoxes(
                    t_boxes[keep], format=t_boxes.format, canvas_size=t_boxes.canvas_size
                )
                target["labels"] = target["labels"][keep]
                target["area"] = target["area"][keep]
                target["iscrowd"] = target["iscrowd"][keep]

        return img, target


def split_to_loader(
    df,
    train_split,
    val_split,
    seed,
    batch_size,
    val_batch_size,
    num_workers,
    dataset_name="default",
    is_distributed=False,
    world_size=1,
    rank=0,
    pin_memory=None,
    prefetch_factor=2,
):
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    g = torch.Generator().manual_seed(seed)
    class_map = new_class_map(sorted(df["Class"].unique().tolist()))
    stems = df["stem"].sort_values().unique().tolist()
    n = len(stems)
    perm = torch.randperm(n, generator=g).tolist()
    stems = [stems[i] for i in perm]

    n_train = int(n * train_split)
    n_val = int(n * val_split)

    train_stems = set(stems[:n_train])
    val_stems = set(stems[n_train: n_train + n_val])
    test_stems = set(stems[n_train + n_val:])

    train_df = df[df["stem"].isin(train_stems)].reset_index(drop=True)
    val_df = df[df["stem"].isin(val_stems)].reset_index(drop=True)
    test_df = df[df["stem"].isin(test_stems)].reset_index(drop=True)

    train_ds = WormDataset(train_df, class_map, transforms=get_transforms(train=True), name=dataset_name)
    val_ds = WormDataset(val_df, class_map, transforms=get_transforms(train=False), name=dataset_name)
    test_ds = WormDataset(test_df, class_map, transforms=get_transforms(train=False), name=dataset_name)

    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=set_worker_seed,
        collate_fn=utils.collate_fn,
        persistent_workers=True if num_workers > 0 else False,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_sampler = None
    val_sampler = None

    if is_distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=seed
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, **loader_kwargs)
        val_loader = DataLoader(val_ds, batch_size=val_batch_size, sampler=val_sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g, **loader_kwargs)
        val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, generator=g, **loader_kwargs)

    test_loader = DataLoader(test_ds, batch_size=val_batch_size, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, train_sampler


def build_dataset_loaders(
    dataset_choice,
    label_path,
    roi_path,
    mask_dir,
    batch_size,
    val_batch_size,
    num_workers,
    train_split,
    val_split,
    seed,
    is_distributed=False,
    world_size=1,
    rank=0,
):
    df = get_full_dataframe(label_path, roi_path, mask_dir)
    multi_df = df[df["Class"] != 4].copy().reset_index(drop=True)
    del df

    if dataset_choice == "multi":
        target_df = multi_df
    elif dataset_choice == "binary":
        target_df = multi_df.copy()
        target_df["Class"] = (target_df["Class"] == 1).astype(int)
    else:
        raise ValueError(f"Unknown dataset choice: {dataset_choice!r}")

    return split_to_loader(
        target_df,
        train_split=train_split,
        val_split=val_split,
        seed=seed,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        dataset_name=dataset_choice,
        is_distributed=is_distributed,
        world_size=world_size,
        rank=rank,
    )