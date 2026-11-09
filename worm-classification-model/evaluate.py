from time import perf_counter
import torch
import torch.distributed as dist
from torch.amp import autocast
import numpy as np
import csv
import os
from pathlib import Path


# ============================================================
# Operating points (Perfect = positive class)
#   argmax   -> Perfect vs rest using argmax (2-class dataset: same as threshold 0.5)
#   p85/p90/p95 ... -> precision-locked: lowest threshold with Perfect precision >= target (max recall)
#   r80/r90/r95 ... -> recall-locked:    threshold with Perfect recall >= target and max precision
#   f1opt    -> threshold with the best Perfect F1 (reference only)
# ============================================================
BIN_METRICS = [
    'acc',
    'prec_perfect', 'recall_perfect', 'f1_perfect',     # Perfect = positive
    'prec_other', 'recall_other', 'f1_other',           # non-Perfect = positive
    'prec_macro', 'recall_macro', 'f1_macro',           # average of the two
    'tp', 'fp', 'fn', 'tn',                             # fp = non-Perfect leaked as Perfect
]


def _pct(t):
    """0.95 -> '95', 0.955 -> '95_5'"""
    return f'{float(t) * 100:g}'.replace('.', '_')


def target_ops(precision_targets=(), recall_targets=()):
    """-> [(name, kind, target), ...] e.g. [('p85', 'precision', 0.85), ..., ('r90', 'recall', 0.9)]"""
    ops = []
    for t in sorted({float(x) for x in precision_targets}):
        ops.append((f'p{_pct(t)}', 'precision', t))
    for t in sorted({float(x) for x in recall_targets}):
        ops.append((f'r{_pct(t)}', 'recall', t))
    return ops


def find_class_index(cls_names, candidates):
    """หา index ของ positive class"""
    # create class index map
    normalized = {name.strip().lower(): i for i, name in enumerate(cls_names)}  # class name มาจาก ImageFolder.classes
    # find positive index
    for cand in candidates:  # alter positive class name
        idx = normalized.get(cand.strip().lower())
        if idx is not None:
            return idx
    return None


def _is_dist():
    return dist.is_available() and dist.is_initialized()


def gather_eval_tensor(t, dataset_len):  # t = tensor model output จาก gpu แต่ละใบ, dataset_len = จำนวนภาพทั้งหมดใน dataset
    """รวม model output จากทุก gpu กลับมา **เรียงตามลำดับ dataset** และตัดภาพที่ sampler วนซ้ำทิ้ง

    DistributedSampler(shuffle=False) แจกภาพแบบสลับกัน (interleave):
        rank 0 -> ภาพ 0, W, 2W, ...   rank 1 -> ภาพ 1, W+1, 2W+1, ...   (W = จำนวน gpu)
    ถ้าหารไม่ลงตัว sampler จะเอาภาพต้น dataset มาเติมท้าย -> ตำแหน่ง >= dataset_len คือภาพซ้ำ

    ดังนั้น local item ที่ j ของ rank r = ภาพ global ตำแหน่ง j * W + r
    -> stack เป็น (max_n, W, ...) แล้ว reshape จะได้ลำดับ 0, 1, 2, ... ตรงกับ dataset.samples
    """
    # check ว่าทำ distributed หรือไม่
    if not _is_dist():
        return t[:dataset_len]  # ส่ง tensor ขนาดเท่าจำนวน dataset กลับไป

    world = dist.get_world_size()   # จำนวน gpu / จำนวน process

    # 1. ถามทุก gpu ว่ามีกี่ภาพ
    local_n = torch.tensor([t.shape[0]], device=t.device, dtype=torch.long)
    sizes = [torch.zeros_like(local_n) for _ in range(world)]
    dist.all_gather(sizes, local_n)
    sizes = [int(s.item()) for s in sizes]
    max_n = max(sizes)

    # 2. pad ให้ทุก gpu มีขนาดเท่ากัน (all_gather ต้องการ shape เดียวกัน)
    if t.shape[0] < max_n:
        pad = torch.zeros((max_n - t.shape[0], *t.shape[1:]), dtype=t.dtype, device=t.device)
        t = torch.cat([t, pad])
    buf = [torch.empty_like(t) for _ in range(world)]
    dist.all_gather(buf, t.contiguous())

    # 3. สลับกลับเป็นลำดับ dataset: แถว j, คอลัมน์ r = ภาพ global j * world + r
    stacked = torch.stack(buf, dim=1)                       # (max_n, world, ...)
    full = stacked.reshape(max_n * world, *t.shape[1:])     # (max_n * world, ...) เรียง 0, 1, 2, ...

    # 4. ทิ้งช่องที่เป็น pad ของเราเอง (rank ที่มีภาพน้อยกว่า max_n)
    valid = torch.zeros((max_n, world), dtype=torch.bool, device=t.device)
    for r, s in enumerate(sizes):
        valid[:s, r] = True
    full = full[valid.reshape(-1)]

    # 5. ตัดภาพที่ sampler วนซ้ำ (อยู่ท้ายสุดเสมอ)
    return full[:dataset_len]


def confusion_matrix(preds, labels, num_classes):
    """นับจำนวนที่ทำนายถูก/ผิด แยกตามคลาส (rows = true, cols = predicted)"""
    idx = labels.long() * num_classes + preds.long()
    return torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def _safe_div(a, b):
    """หารอย่างปลอดภัย ป้องกันหารด้วยศูนย์"""
    return torch.where(b > 0, a / b.clamp_min(1e-12), torch.zeros_like(a))


def metrics_from_cm(cm):
    """All classes (argmax): Accuracy + macro Precision / Recall / F1"""
    cm = cm.double()
    tp = cm.diag()
    prec = _safe_div(tp, cm.sum(0))
    rec = _safe_div(tp, cm.sum(1))
    f1 = _safe_div(2 * prec * rec, prec + rec)
    return {
        'acc': (tp.sum() / cm.sum().clamp_min(1)).item(),
        'prec_macro': prec.mean().item(),
        'recall_macro': rec.mean().item(),
        'f1_macro': f1.mean().item(),
    }


def _prf(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return prec, rec, f1


def binary_block(pred_pos, true_pos):
    """Perfect vs rest at one operating point.
    Returns (metrics dict with BIN_METRICS keys, 2x2 cm [[TP, FN], [FP, TN]] rows = true)."""
    tp = int((pred_pos & true_pos).sum())
    fp = int((pred_pos & ~true_pos).sum())
    fn = int((~pred_pos & true_pos).sum())
    tn = int((~pred_pos & ~true_pos).sum())
    total = tp + fp + fn + tn

    p_pos, r_pos, f_pos = _prf(tp, fp, fn)   # Perfect as positive
    p_neg, r_neg, f_neg = _prf(tn, fn, fp)   # non-Perfect as positive (roles swapped)

    m = {
        'acc': (tp + tn) / total if total else 0.0,
        'prec_perfect': p_pos, 'recall_perfect': r_pos, 'f1_perfect': f_pos,
        'prec_other': p_neg, 'recall_other': r_neg, 'f1_other': f_neg,
        'prec_macro': (p_pos + p_neg) / 2, 'recall_macro': (r_pos + r_neg) / 2, 'f1_macro': (f_pos + f_neg) / 2,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }
    cm = torch.tensor([[tp, fn], [fp, tn]], dtype=torch.long)
    return m, cm


def by_class_block(pred_pos, labels, num_classes):
    """Multi-class breakdown at one operating point: rows = every TRUE class, cols = [Perfect, Others].
    -> shows WHICH non-Perfect class leaks into the Perfect basket at this threshold.
    shape (num_classes, 2): [:, 0] = picked as Perfect, [:, 1] = rejected (Others)"""
    idx = labels.long() * 2 + (~pred_pos).long()
    return torch.bincount(idx, minlength=num_classes * 2).reshape(num_classes, 2).cpu()


# ============================================================
# Exact threshold search (every unique P(Perfect) on the split, no 0.01 grid)
# ============================================================
def threshold_curve(p_pos, true_pos):
    """Precision / recall / F1 for EVERY possible cut 'P(Perfect) >= thr'.
    thr runs over the unique probabilities, highest first -> recall only goes up along the curve."""
    p = p_pos.detach().double().flatten()
    y = true_pos.detach().bool().flatten()
    if p.numel() == 0:
        return None
    n_pos = int(y.sum())
    n_neg = int(p.numel() - n_pos)

    order = torch.argsort(p, descending=True)
    p_s = p[order]
    y_s = y[order].double()
    tp = torch.cumsum(y_s, 0)
    fp = torch.cumsum(1.0 - y_s, 0)

    # tied probabilities must be cut together -> keep only the last index of each tie group
    last = torch.ones_like(p_s, dtype=torch.bool)
    if p_s.numel() > 1:
        last[:-1] = p_s[:-1] != p_s[1:]
    thr, tp, fp = p_s[last], tp[last], fp[last]

    prec = tp / (tp + fp)                                    # tp + fp >= 1 always
    rec = tp / n_pos if n_pos else torch.zeros_like(tp)
    f1 = 2 * tp / (2 * tp + fp + (n_pos - tp)).clamp_min(1e-12)
    return {'thr': thr, 'prec': prec, 'rec': rec, 'f1': f1, 'tp': tp, 'fp': fp,
            'n_pos': n_pos, 'n_neg': n_neg}


def average_precision(curve):
    """PR-AUC (sklearn definition: sum of (R_n - R_{n-1}) * P_n). Threshold-free."""
    if curve is None or curve['n_pos'] == 0:
        return None
    rec = curve['rec']
    d_rec = torch.diff(rec, prepend=rec.new_zeros(1))
    return float((d_rec * curve['prec']).sum())


def roc_auc(curve):
    """ROC-AUC from the same curve (trapezoid). Threshold-free."""
    if curve is None or curve['n_pos'] == 0 or curve['n_neg'] == 0:
        return None
    zero = curve['rec'].new_zeros(1)
    tpr = torch.cat([zero, curve['rec']])
    fpr = torch.cat([zero, curve['fp'] / curve['n_neg']])
    return float(torch.trapezoid(tpr, fpr))


def pick_precision_threshold(curve, target):
    """Precision-locked: lowest threshold whose precision >= target
    (= the deepest cut that still keeps the basket clean -> highest recall).
    If no cut reaches the target -> the cut with the highest precision (largest recall among ties)."""
    feasible = curve['prec'] >= target - 1e-9
    if bool(feasible.any()):
        idx = int(feasible.nonzero()[-1])
    else:
        pmax = curve['prec'].max()
        idx = int((curve['prec'] >= pmax - 1e-12).nonzero()[-1])
    return float(curve['thr'][idx])


def pick_recall_threshold(curve, target):
    """Recall-locked: among cuts with recall >= target, the one with the highest precision
    (ties -> the higher threshold = fewer picks). If none reach it -> take everything (last cut)."""
    feasible = curve['rec'] >= target - 1e-9
    if bool(feasible.any()):
        idx_f = feasible.nonzero().squeeze(1)
        idx = int(idx_f[int(torch.argmax(curve['prec'][idx_f]))])   # argmax returns the first max
    else:
        idx = curve['thr'].numel() - 1
    return float(curve['thr'][idx])


def pick_f1_threshold(curve):
    """Threshold with the best Perfect F1 (reference only)."""
    return float(curve['thr'][int(torch.argmax(curve['f1']))])


def _curve_for_plot(curve, max_points=5000):
    """Small CPU numpy copy of the PR curve (for the PR-curve PNG)."""
    rec = curve['rec'].detach().cpu().numpy()
    prec = curve['prec'].detach().cpu().numpy()
    if rec.size > max_points:
        idx = np.unique(np.linspace(0, rec.size - 1, max_points).astype(np.int64))
        rec, prec = rec[idx], prec[idx]
    return {'rec': rec, 'prec': prec}


def compute_metrics(probs, labels, num_classes, pos_idx=None, ops=(), thresholds=None):
    """All metrics from softmax probabilities.

    Keys (always):
      loss, all_acc, all_prec_macro, all_recall_macro, all_f1_macro, cm      -> all classes, argmax
    Keys (when a Perfect class exists):
      ap, roc_auc                                      -> threshold-free
      argmax_<BIN_METRICS>, cm_argmax                  -> Perfect vs rest, argmax
      <op>_threshold, <op>_met, <op>_<BIN_METRICS>, cm_<op>   for every op in `ops` (p85, r90, ...)
      f1opt_threshold, f1opt_<BIN_METRICS>, cm_f1opt   -> best-F1 threshold (reference)
      cm_<op>_by_class                                 -> (num_classes, 2) rows = true class, cols = [Perfect, Others]
                                                          (only when num_classes > 2)
      curve                                            -> {'rec', 'prec'} numpy, for the PR plot

    ops        = evaluate.target_ops(...)
    thresholds = None            -> search every threshold on THIS split (use on val)
                 {op_name: t}    -> apply frozen val thresholds (use on test), must include 'f1opt'
    """
    out = {}
    labels = labels.long()
    p_true = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    out['loss'] = (-torch.log(p_true.clamp_min(1e-12))).mean().item()  # = CrossEntropy

    preds = probs.argmax(1)
    cm = confusion_matrix(preds, labels, num_classes)
    out.update({f'all_{k}': v for k, v in metrics_from_cm(cm).items()})
    out['cm'] = cm.cpu()

    if pos_idx is None:
        return out

    true_pos = labels == pos_idx
    p_pos = probs[:, pos_idx]

    # threshold-free
    curve = threshold_curve(p_pos, true_pos)
    out['ap'] = average_precision(curve)
    out['roc_auc'] = roc_auc(curve)
    out['curve'] = _curve_for_plot(curve) if curve is not None else None

    # argmax operating point
    b, bcm = binary_block(preds == pos_idx, true_pos)
    out.update({f'argmax_{k}': v for k, v in b.items()})
    out['cm_argmax'] = bcm

    # thresholds: search (val) or frozen (test)
    if thresholds is None:
        thr = {}
        for name, kind, target in ops:
            if curve is None:
                thr[name] = 0.5
            elif kind == 'precision':
                thr[name] = pick_precision_threshold(curve, target)
            else:
                thr[name] = pick_recall_threshold(curve, target)
        thr['f1opt'] = pick_f1_threshold(curve) if curve is not None else 0.5
    else:
        thr = {k: float(v) for k, v in thresholds.items()}

    for name, kind, target in list(ops) + [('f1opt', None, None)]:
        t = thr[name]
        pred_pos = p_pos >= t
        b, bcm = binary_block(pred_pos, true_pos)
        out.update({f'{name}_{k}': v for k, v in b.items()})
        out[f'{name}_threshold'] = float(t)
        out[f'cm_{name}'] = bcm
        if num_classes > 2:
            out[f'cm_{name}_by_class'] = by_class_block(pred_pos, labels, num_classes)
        if kind == 'precision':
            out[f'{name}_met'] = int(b['prec_perfect'] >= target - 1e-9)
        elif kind == 'recall':
            out[f'{name}_met'] = int(b['recall_perfect'] >= target - 1e-9)

    return out


@torch.no_grad()
def benchmark_inference(model, img_size, device, amp_enabled=False, amp_dtype=torch.float16,
                        batch_size=64, warmup=20, iters=100):
    """Model forward time only (no data loading / preprocessing), synthetic input.

    inference_latency_ms : time per image at batch size 1
    inference_fps        : images per second at batch_size
    """
    model.eval()
    is_cuda = device.type == 'cuda'

    def sync():
        if is_cuda:
            torch.cuda.synchronize(device)

    def run(x, n):
        for _ in range(n):
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                model(x)

    def timed(x, n):
        sync()
        t0 = perf_counter()
        run(x, n)
        sync()
        return perf_counter() - t0

    x1 = torch.randn(1, 3, img_size, img_size, device=device)
    run(x1, warmup)
    latency_ms = timed(x1, iters) / iters * 1000

    xb = torch.randn(batch_size, 3, img_size, img_size, device=device)
    n_b = max(10, iters // 4)
    run(xb, max(3, warmup // 4))
    fps = batch_size * n_b / timed(xb, n_b)

    return {'inference_latency_ms': latency_ms, 'inference_fps': fps, 'inference_batch_size': batch_size}


def save_predictions(probs, labels, samples, class_names, out_path, pos_idx=None, thresholds=None):
    """เขียน CSV 1 แถวต่อ 1 ภาพ (เรียงตาม dataset)
    samples    = ImageFolder.samples -> [(path, class_idx), ...]
    thresholds = {op_name: t} (e.g. {'p95': 0.96, 'r90': 0.31}) -> one pred_/outcome_ column pair per op
    outcome (Perfect = positive, ตัดสินด้วย P(Perfect) >= threshold):
      TP = Perfect -> Perfect | FP = Other -> Perfect | FN = Perfect -> Other | TN = Other -> Other
    returns {'n': N, 'FP_<op>': ..., 'FN_<op>': ...}
    """
    probs = probs.detach().float().cpu()
    labels = labels.detach().long().cpu()
    n = probs.shape[0]
    if len(samples) != n:
        raise ValueError(f"save_predictions: {len(samples)} image paths but {n} predictions")
    # check ว่าลำดับ prediction ตรงกับลำดับภาพ
    mismatch = int((torch.tensor([s[1] for s in samples]) != labels).sum())
    if mismatch:
        raise ValueError(f"save_predictions: {mismatch} labels do not match the image list -> order is wrong")

    preds = probs.argmax(1)
    thresholds = dict(thresholds or {})
    has_bin = pos_idx is not None and bool(thresholds)
    counts = {'n': n}
    if has_bin:
        pos_name, neg_name = class_names[pos_idx], 'Others'
        true_pos = labels == pos_idx
        pred_pos = {op: probs[:, pos_idx] >= t for op, t in thresholds.items()}
        for op in thresholds:
            counts[f'FP_{op}'] = 0
            counts[f'FN_{op}'] = 0

    fields = ['index', 'path', 'filename', 'true_label', 'pred_label', 'correct', 'confidence']
    fields += [f'p_{name}' for name in class_names]
    if has_bin:
        fields += ['true_binary']
        for op in thresholds:
            fields += [f'pred_{op}', f'outcome_{op}']

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f'{out_path.name}.tmp{os.getpid()}')
    with open(tmp_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(n):
            t, p = int(labels[i]), int(preds[i])
            row = {'index': i, 'path': samples[i][0], 'filename': Path(samples[i][0]).name,
                   'true_label': class_names[t], 'pred_label': class_names[p], 'correct': int(t == p),
                   'confidence': round(float(probs[i].max()), 6)}
            for j, name in enumerate(class_names):
                row[f'p_{name}'] = round(float(probs[i, j]), 6)
            if has_bin:
                tp_ = bool(true_pos[i])
                row['true_binary'] = pos_name if tp_ else neg_name
                for op in thresholds:
                    pp_ = bool(pred_pos[op][i])
                    outcome = ('TP' if pp_ else 'FN') if tp_ else ('FP' if pp_ else 'TN')
                    if outcome in ('FP', 'FN'):
                        counts[f'{outcome}_{op}'] += 1
                    row[f'pred_{op}'] = pos_name if pp_ else neg_name
                    row[f'outcome_{op}'] = outcome
            writer.writerow(row)
    os.replace(tmp_path, out_path)   # atomic
    return counts