"""Error galleries for trained models (TP / FP / FN / TN).

อ่าน   OUTPUT_DIR/<experiment>/pred/<dataset>/<family>/<variant>/<split>_predictions.csv
       (training เขียนให้อัตโนมัติ) ถ้าไม่มีไฟล์ (model เทรนก่อนมี feature นี้) หรือใส่ --repredict
       จะโหลด best.pt + best_meta.json แล้ว predict ใหม่

เขียน  OUTPUT_DIR/<experiment>/pred/<dataset>/<family>/<variant>/analysis/<split>/
         fp_gallery.png        FP ที่ model มั่นใจที่สุด  (non-Perfect ถูกทายเป็น Perfect)
         fn_gallery.png        FN ที่ model มั่นใจที่สุด  (Perfect ถูกทายเป็น non-Perfect)
         outcome_summary.csv   จำนวน TP / FP / FN / TN และ %
       OUTPUT_DIR/<experiment>/pred/<dataset>/all_models_<split>_outcomes.csv   (1 แถวต่อ 1 model)

ใช้
  python analyze.py -ds 2class_blackout -m resnet              # ทุก resnet, test set
  python analyze.py -ds 2class_blackout -m all --split val     # ทุก model, val set
  python analyze.py -ds 5class_raw -m vit_b_16 --repredict
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import config as c


def find_models(exp_dir, dataset, model):
    """(family_dir, variant, ckpt_dir) ของทุก model ที่มี best_meta.json"""
    found = []
    for meta_path in sorted((exp_dir / 'ckpt' / dataset).glob('*/*/best_meta.json')):
        variant = meta_path.parent.name
        m = model.lower()
        if m == 'all' or variant == m or variant.startswith(m):
            found.append((meta_path.parent.parent.name, variant, meta_path.parent))
    return found


def repredict(variant, ckpt_dir, dataset, split, out_csv):
    """predict ใหม่จาก best.pt (ต้องใช้ torch, ควรมี GPU)"""
    from functools import partial

    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    from data import worm_transforms
    from evaluate import find_class_index, save_predictions
    from models import build_model

    meta = json.loads((ckpt_dir / 'best_meta.json').read_text())
    pt_path = ckpt_dir / 'best.pt'
    if not pt_path.exists():
        raise FileNotFoundError(f'{pt_path} not found (training not finished?)')

    tf = partial(worm_transforms, img_size=meta.get('img_size', c.IMG_SIZE),
                 upscale=meta.get('upscale', c.UPSCALE), augment=False)
    ds = ImageFolder(Path(c.DATASETS[dataset]) / split, transform=tf)
    class_names = meta.get('class_names') or ds.classes
    if ds.classes != class_names:
        raise ValueError(f'class folders {ds.classes} != classes used in training {class_names}')

    device = c.DEVICE
    amp_on = device.type == 'cuda' and c.USE_AMP
    amp_dtype = torch.bfloat16 if (amp_on and c.AMP_DTYPE == 'bf16' and torch.cuda.is_bf16_supported()) \
        else torch.float16

    model = build_model(variant, len(class_names), finetune_path=pt_path, device=device)
    model.eval()
    loader = DataLoader(ds, batch_size=c.BATCH_SIZE * 2, shuffle=False,
                        num_workers=c.NUM_WORKERS, pin_memory=c.PIN_MEMORY)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_on):
                out = model(imgs.to(device, non_blocking=True))
            all_probs.append(torch.softmax(out.float(), dim=1).cpu())
            all_labels.append(labels)

    pos_idx = meta.get('positive_index')
    if pos_idx is None:
        pos_idx = find_class_index(class_names, c.POSITIVE_CLASS_NAMES)
    counts = save_predictions(torch.cat(all_probs), torch.cat(all_labels), ds.samples, class_names,
                              out_csv, pos_idx=pos_idx, threshold=meta.get('threshold'))
    print(f'   📝 re-predicted {split}: {counts}')


def _preview(path, img_size):
    """แสดงภาพแบบที่ model เห็น: วางกลาง canvas ดำ, ย่อเฉพาะภาพที่ใหญ่กว่า img_size"""
    try:
        img = Image.open(path).convert('RGB')
    except Exception:
        return np.zeros((img_size, img_size, 3), dtype=np.uint8)
    if max(img.size) > img_size:
        img.thumbnail((img_size, img_size))
    canvas = Image.new('RGB', (img_size, img_size))
    canvas.paste(img, ((img_size - img.width) // 2, (img_size - img.height) // 2))
    return np.asarray(canvas)


def make_gallery(df, title, out_path, p_col, img_size, cols=c.GALLERY_COLS):
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.axis('off')
        ax.text(0.5, 0.5, f'{title}\n(no images)', ha='center', va='center')
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        return

    rows = math.ceil(len(df) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.9, rows * 2.25 + 0.6))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis('off')
    for ax, (_, r) in zip(axes, df.iterrows()):
        ax.imshow(_preview(r['path'], img_size))
        ax.set_title(f"{Path(r['filename']).stem}\n{r['true_label']} | p={r[p_col]:.2f}", fontsize=6.5)
    fig.suptitle(title, fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def analyze_one(pred_csv, out_dir, meta, variant, family_dir, split, k):
    df = pd.read_csv(pred_csv)
    if 'outcome' not in df.columns:
        print('   no Perfect class / threshold in this dataset -> skipped')
        return None
    out_dir.mkdir(parents=True, exist_ok=True)

    pos_name = meta.get('positive_class') or c.POSITIVE_CLASS_NAMES[0]
    p_col = f'p_{pos_name}'
    thr = float(df['threshold'].iloc[0])
    img_size = meta.get('img_size', c.IMG_SIZE)
    tag = f'{variant} | {split}'

    cnt = df['outcome'].value_counts().reindex(['TP', 'FP', 'FN', 'TN'], fill_value=0)
    tp, fp, fn, tn = (int(cnt[o]) for o in ('TP', 'FP', 'FN', 'TN'))
    meaning = {'TP': f'{pos_name} -> {pos_name}', 'FP': f'non-{pos_name} -> {pos_name}',
               'FN': f'{pos_name} -> non-{pos_name}', 'TN': f'non-{pos_name} -> non-{pos_name}'}
    pd.DataFrame({'outcome': cnt.index, 'meaning': [meaning[o] for o in cnt.index],
                  'count': cnt.values, 'pct_of_all': (cnt.values / len(df) * 100).round(2)}) \
        .to_csv(out_dir / 'outcome_summary.csv', index=False)

    fp_df = df[df['outcome'] == 'FP'].sort_values(p_col, ascending=False).head(k)   # มั่นใจผิดมากสุด
    fn_df = df[df['outcome'] == 'FN'].sort_values(p_col, ascending=True).head(k)
    make_gallery(fp_df, f'{tag} | most confident FP: non-{pos_name} predicted {pos_name} '
                        f'({fp} total, showing {len(fp_df)})', out_dir / 'fp_gallery.png', p_col, img_size)
    make_gallery(fn_df, f'{tag} | most confident FN: {pos_name} predicted non-{pos_name} '
                        f'({fn} total, showing {len(fn_df)})', out_dir / 'fn_gallery.png', p_col, img_size)

    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    print(f'   TP {tp} | FP {fp} | FN {fn} | TN {tn} | P {prec:.4f} R {rec:.4f} F1 {f1:.4f} @thr {thr:.2f}')
    return {'variant': variant, 'family': family_dir, 'split': split, 'n': len(df), 'threshold': thr,
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 'precision': prec, 'recall': rec, 'f1': f1}


def main():
    ap = argparse.ArgumentParser(description='TP / FP / FN / TN galleries for trained models')
    ap.add_argument('--dataset', '-ds', required=True, choices=list(c.DATASETS.keys()))
    ap.add_argument('--model', '-m', required=True, help="variant (resnet18), family (resnet) or 'all'")
    ap.add_argument('--experiment', '-e', default=c.EXPERIMENT_NAME)
    ap.add_argument('--split', default='test', choices=['test', 'val'])
    ap.add_argument('--k', type=int, default=c.GALLERY_K, help='images per gallery')
    ap.add_argument('--repredict', action='store_true',
                    help='predict again from best.pt (otherwise only when the CSV is missing)')
    args = ap.parse_args()

    exp_dir = Path(c.OUTPUT_DIR) / args.experiment
    models = find_models(exp_dir, args.dataset, args.model)
    if not models:
        raise SystemExit(f"No trained models matching '{args.model}' in {exp_dir / 'ckpt' / args.dataset}")

    rows = []
    for family_dir, variant, ckpt_dir in models:
        print(f'🔎 {variant} ({family_dir})')
        meta = json.loads((ckpt_dir / 'best_meta.json').read_text())
        var_dir = exp_dir / 'pred' / args.dataset / family_dir / variant
        pred_csv = var_dir / f'{args.split}_predictions.csv'
        try:
            if args.repredict or not pred_csv.exists():
                repredict(variant, ckpt_dir, args.dataset, args.split, pred_csv)
            row = analyze_one(pred_csv, var_dir / 'analysis' / args.split, meta, variant, family_dir,
                              args.split, args.k)
            if row:
                rows.append(row)
                print(f'   ✅ {var_dir / "analysis" / args.split}')
        except Exception as e:
            print(f'   ❌ {variant}: {e}')

    if rows:
        out = exp_dir / 'pred' / args.dataset / f'all_models_{args.split}_outcomes.csv'
        new = pd.DataFrame(rows)
        if out.exists():   # เก็บผลของ model อื่นที่ไม่ได้ run รอบนี้ไว้
            old = pd.read_csv(out)
            new = pd.concat([old[~old['variant'].isin(new['variant'])], new], ignore_index=True)
        new.sort_values('f1', ascending=False).to_csv(out, index=False)
        print(f'\n📊 All models ({args.split}): {out}')


if __name__ == '__main__':
    main()