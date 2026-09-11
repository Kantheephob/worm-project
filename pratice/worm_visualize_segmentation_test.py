import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
from PIL import Image
from collections import Counter
from pathlib import Path

classes_map = {
    0: 'Young',
    1: 'Perfect',
    2: 'Old',
    3: 'Undecidable',
    4: 'Not worm',
    5: 'Error / Miss segmentation',
    6: 'Low quality',
}

color_map = {
    0: 'Red',
    1: 'Green',
    2: 'Blue',
    3: 'Orange',
    4: 'Yellow',
    5: 'Cyan',
    6: 'Purple',  
}

def get_dataset(labels_path, roi_metadata_path, masks_dir):
    """
    Merge label CSV with ROI metadata and attach per-instance mask info
    (V, L, A, BY, bUse, roi_bbox_xyxy, max_ratio, boundary, img_h, img_w)
    by joining each image's rows against its mask pickle on bbox.

    Parameters
    ----------
    labels_path : str or Path
        CSV with 'Filename' (format: 'stem_idx.npy') and 'Class' columns.
    roi_metadata_path : str or Path
        Pickle with columns 'stem', 'idx', 'bbox', and other ROI info.
    masks_dir : str or Path
        Directory containing per-stem mask pickle files ('{stem}.pkl'),
        each with a 'bbox' column matching the ROI metadata's bbox.

    Returns
    -------
    pd.DataFrame
        Fully merged dataframe, one row per instance.
    """
    labels = pd.read_csv(labels_path)
    labels[['stem', 'idx']] = labels['Filename'].str.extract(r'(.*)_(\d+)\.npy')
    labels['idx'] = labels['idx'].astype(int)

    metadata = pd.read_pickle(roi_metadata_path)

    join_df = pd.merge(labels, metadata, on=['stem', 'idx'], how='inner').sort_values('stem')
    join_df['mask_path'] = join_df['stem'].apply(lambda s: str(Path(masks_dir) / f'{s}.pkl'))
    join_df['bbox_key'] = join_df['bbox'].apply(lambda x: tuple(int(np.round(float(val))) for val in x))
    
    merged_chunks = []
    for stem, df_img in join_df.groupby('stem', sort=False):
        df_img = df_img.copy()
        df_img.drop_duplicates(subset=['bbox_key'], keep='first', inplace=True)
        
        mask_path = df_img['mask_path'].iloc[0]

        masks_df = pd.read_pickle(mask_path).copy()
        masks_df['bbox'] = masks_df['bbox'].apply(tuple)
        masks_df['bbox_key'] = masks_df['bbox'].apply(lambda x: tuple(int(np.round(float(val))) for val in x))

        if 'predicted_iou' in masks_df.columns:
            masks_df = masks_df.sort_values('predicted_iou', ascending=False)

        masks_df = masks_df.drop_duplicates(subset=['bbox_key'], keep='first')
        
        merged = pd.merge(df_img, masks_df, on='bbox_key', how='inner', suffixes=('', '_mask'))
        merged_chunks.append(merged)

    full_df = pd.concat(merged_chunks, ignore_index=True)
    return full_df

def show_bbox_with_class(bbox, ax, label): # XYWH
    """
    Draw a bounding box with a class label on a matplotlib Axes.

    Parameters
    ----------
    bbox : tuple or list of (x, y, w, h)
        Bounding box in XYWH format (top-left corner + width/height).
    ax : matplotlib.axes.Axes
        The axes to draw on.
    label : int
        Class ID used to look up color from `color_map` and display as text.
    """
    x, y, w, h = bbox

    box_color = color_map.get(label)
    ax.add_patch(plt.Rectangle((x, y), w, h, edgecolor=box_color, facecolor=(0, 0, 0, 0), lw=2)) # box color
    ax.text((x + (x + w)) // 2, (y + (y + h)) // 2, # position
            f"{label}", color='white', fontsize=12, weight='bold', # text beauty
            ha='center', va='center', # set item center
            # bbox=dict(facecolor=box_color, edgecolor='none', alpha=0.5, pad=3) # text background
           )

def show_mask_with_class(mask_2d, ax, offset_x, offset_y, block_h, block_w, cls, alpha=0.4):
    """
    Overlay a segmentation mask (cropped to the current block) on a matplotlib Axes.

    The mask is in global image coordinates and is cropped to the block region
    defined by (offset_x, offset_y, block_w, block_h) before being displayed.

    Parameters
    ----------
    mask_2d : array-like of bool
        Full-image binary segmentation mask (H x W).
    ax : matplotlib.axes.Axes
        The axes to draw on.
    offset_x : int
        Horizontal offset of the block in global image coordinates (pixels from left).
    offset_y : int
        Vertical offset of the block in global image coordinates (pixels from top).
    block_h : int
        Height of the current block in pixels.
    block_w : int
        Width of the current block in pixels.
    cls : int
        Class ID used to look up color from `color_map`.
    alpha : float, optional
        Opacity of the mask overlay. Default is 0.4.
    """
    # Get the color string, defaulting to 'white' if class is unknown
    color_name = color_map.get(cls, 'white')  
    
    # Convert the string (e.g., 'cyan') to an RGB tuple (e.g., (0.0, 1.0, 1.0))
    rgb_color = mcolors.to_rgb(color_name) 

    mask_2d = np.array(mask_2d, dtype=bool)

    # crop mask จาก global → local block
    local_mask = mask_2d[offset_y : offset_y + block_h,
                         offset_x : offset_x + block_w]

    overlay = np.zeros((*local_mask.shape, 4), dtype=np.float32)
    
    # Now this unpacks the numeric tuple correctly: [0.0, 1.0, 1.0, 0.4]
    overlay[local_mask] = [*rgb_color, alpha] 

    ax.imshow(overlay)

def plot_segmentation_blocks(label_path, roi_metadata_path, pkl_dir, start=1, stop=1, block_h=None, block_w=None, show_bbox=True, show_mask=False):
    """
    Visualize worm segmentation results by splitting each image into 4 overlapping
    corner blocks and overlaying bounding boxes and/or masks per block.

    For each image in the specified range, the function merges label and ROI metadata,
    loads per-image segmentation data from a pickle file, assigns each worm ROI to
    whichever corner blocks it overlaps, and plots each block as a separate figure
    with a class summary caption.

    Parameters
    ----------
    label_path : str or Path
        Path to the CSV file containing worm class labels.
        Expected columns: 'Filename' (format: 'worm-000001_000.npy'), 'Class'.

    roi_metadata_path : str or Path
        Path to the pickle file containing ROI metadata (bounding boxes, image paths).
        Expected columns: 'stem', 'idx', 'bbox', 'raw_img_path'.

    pkl_dir : str or Path
        Directory containing per-image pickle files (e.g., 'image_stem.pkl')
        with segmentation masks and bounding boxes.

    start : int, optional
        1-based index of the first image to display. Default is 1.

    stop : int, optional
        1-based index of the last image to display (inclusive). Default is 1.

    block_h : int or None, optional
        Height of each corner block in pixels.
        Must be between image_h // 2 and image_h.
        Defaults to image_h // 2 if None or out of range.

    block_w : int or None, optional
        Width of each corner block in pixels.
        Must be between image_w // 2 and image_w.
        Defaults to image_w // 2 if None or out of range.

    show_bbox : bool, optional
        If True, draws bounding boxes on each block. Default is True.

    show_mask : bool, optional
        If True, overlays segmentation masks on each block and uses
        color names in the summary caption. Default is False.

    Notes
    -----
    - Classes 4 (Not worm) and 5 (Error/Miss segmentation) are excluded from display.
    - ROIs are assigned to a block if their bounding box overlaps the block's region.
      A single ROI may appear in multiple blocks if it spans a corner boundary.
    - Each block is plotted as a separate 16x9 figure with a title and class summary.
    - Requires: pandas, numpy, PIL, matplotlib, pathlib, collections.Counter.
    """
    
    # เตรียม dataframe
    df = get_dataset(label_path, roi_metadata_path, pkl_dir)
    
    # ดึงรูปทั้งหมด
    image_list_all = df['raw_img_path'].drop_duplicates().sort_values().tolist()
    total_images = len(image_list_all)

    # validate input
    if start < 1 or stop < 1:
        print(f"Error: 'start' and 'stop' must be >= 1. (Got start={start}, stop={stop}). Defaulting to 1.")
        start = max(1, start)
        stop = max(1, stop)
        
    if start > total_images:
        print(f"Error: 'start' ({start}) exceeds total images ({total_images}). Defaulting start to {total_images}.")
        start = total_images
        
    if stop > total_images:
        print(f"Error: 'stop' ({stop}) exceeds total images ({total_images}). Defaulting stop to {total_images}.")
        stop = total_images

    if start > stop:
        print(f"Error: 'start' ({start}) > 'stop' ({stop}). Defaulting stop to match start.")
        stop = start

    image_list = image_list_all[start-1 : stop]
    
    # ทำงานกับทีละรูป
    for i, image in enumerate(image_list):
        stem = Path(image).stem # ได้ชื่อไฟล์ไม่มีนามสกุล

        # โหลด pkl
        pkl_path = Path(pkl_dir) / f'{stem}.pkl'
        if not pkl_path.exists():
            print(f'Warning: {pkl_path} not found, skipping')
            continue
        pkl_df = pd.read_pickle(pkl_path)

        # ดึงแถวข้อมูลที่เกี่ยวข้องกับภาพ
        image_row = df[(df['raw_img_path'] == image) & (~df['Class'].isin([4, 5]))]

        image_row['bbox'] = image_row['bbox'].apply(tuple)
        pkl_df['bbox'] = pkl_df['bbox'].apply(tuple)
        
        image_df = pd.merge(image_row, pkl_df, on='bbox', how='inner')
        
        # ดึง features ที่ต้องใช้
        bboxes = image_df['bbox']
        classes = image_df['Class']
        masks = image_df['segmentation']
        
        image = np.array(Image.open(image).convert('RGB'))
        image_h, image_w = image.shape[:2]
    
        middle_h, middle_w = image_h // 2, image_w // 2
        #  validate block size
        block_h = block_h if (block_h is not None and middle_h <= block_h <= image_h) else middle_h
        block_w = block_w if (block_w is not None and middle_w <= block_w <= image_w) else middle_w
        
        # plot box structure
        blocks = {
            'top_left': {
                'img': image[:block_h, :block_w], 
                'offset': (0, 0), 'bboxes': [], 'classes': [], 'masks': [] 
            },
            'top_right': {
                'img': image[:block_h, image_w - block_w:], 
                'offset': (image_w - block_w, 0), 'bboxes': [], 'classes': [], 'masks': []  
            },
            'bottom_left': {
                'img': image[image_h - block_h:, :block_w], 
                'offset': (0, image_h - block_h), 'bboxes': [], 'classes': [], 'masks': []  
            },
            'bottom_right': {
                'img': image[image_h - block_h:, image_w - block_w:], 
                'offset': (image_w - block_w, image_h - block_h), 'bboxes': [], 'classes': [], 'masks': []  
            },
        }
        
        # plot bbox and class
        for bbox, cls, mask in zip(bboxes, classes, masks):
            x, y, w, h = bbox
            x_max, y_max = x + w, y + h
            
            overlaps = {
                'top_left': (
                    x < block_w and
                    y < block_h
                ),
                'top_right': (
                    x_max > image_w - block_w and
                    y < block_h
                ),
                'bottom_left': (
                    x < block_w and
                    y_max > image_h - block_h
                ),
                'bottom_right': (
                    x_max > image_w - block_w and
                    y_max > image_h - block_h
                )
            }

            # overlaps = {
            #     'top_left': (
            #         x < middle_w and
            #         y < middle_h
            #     ),
            #     'top_right': (
            #         x_max > middle_w and 
            #         y < middle_h
            #     ),
            #     'bottom_left': (
            #         x < middle_w and 
            #         y_max > middle_h
            #     ),
            #     'bottom_right': (
            #         x_max > middle_w and 
            #         y_max > middle_h
            #     )
            # }

            # assign block 
            for block_key, is_overlap in overlaps.items():
                if is_overlap:
                    offset_x, offset_y = blocks[block_key]['offset']
                    local_bbox = [x - offset_x, y - offset_y, w, h]
                    blocks[block_key]['bboxes'].append(local_bbox)
                    blocks[block_key]['classes'].append(cls)
                    blocks[block_key]['masks'].append(mask)

        # draw block
        img_num = start + i
        for idx, (block_key, values) in enumerate(blocks.items()):
            block_img = values['img']
            block_bboxes = values['bboxes']
            block_classes = values['classes']
            block_masks = values['masks']
            offset_x, offset_y = values['offset']
            bh_actual, bw_actual = block_img.shape[:2]

            counter = Counter(block_classes)
            if show_mask:
                summary = ' | '.join(  
                    f"{color_map[cls_id]} -> {classes_map[cls_id]} = {count}"
                    for cls_id, count in sorted(counter.items())
                )
            else:
                summary = ' | '.join(  
                    f"{cls_id} -> {classes_map[cls_id]} = {count}"
                    for cls_id, count in sorted(counter.items())
                )

            fig, ax = plt.subplots(figsize=(16, 9))
            if idx == 0:
                fig.suptitle(f'Image {img_num} | Total: {len(masks)} objects', fontsize=24, fontweight='bold', ha='center')
                
            ax.set_title(f'{stem} — {block_key}: {len(block_bboxes)} objects', fontsize=18, loc='center')
            fig.text(0.5, -0.05, f'{summary}', ha='center', va='top', transform=ax.transAxes, fontsize=14)
            ax.imshow(block_img)
            ax.axis('off')

            if show_bbox:
                for bbox, cls in zip(block_bboxes, block_classes):
                    show_bbox_with_class(bbox, ax, cls)

            if show_mask:
                for mask, cls in zip(block_masks, block_classes):
                    if mask is None:
                        continue
                    show_mask_with_class(mask, ax, offset_x, offset_y, bh_actual, bw_actual, cls)

            ax.set_xlim(0, bw_actual)
            ax.set_ylim(bh_actual, 0)
            fig.subplots_adjust(bottom=0.2)
            plt.tight_layout(pad=1.5)
            plt.show()