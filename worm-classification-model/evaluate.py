from time import perf_counter
import torch
import torch.distributed as dist
from torch.amp import autocast
import csv
import os
from pathlib import Path


def find_class_index(cls_names, candidates):
    """หา index ของ positive class"""
    # create class index map
    normalized = {name.strip().lower(): i for i, name in enumerate(cls_names)}  # class name มาจาก ImageFolder.classes
    # find positive index 
    for cand in candidates: # alter positive class name
        idx = normalized.get(cand.strip().lower())
        if idx is not None:
            return idx
    return None


def _is_dist():
    return dist.is_available() and dist.is_initialized()


def gather_eval_tensor(t, dataset_len): # t = tensor model output จาก gpu แต่ละใบ, dataset_len = จำนวนภาพทั้งหมดใน dataset
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
    """นับจำนวนที่ทำนายถูก/ผิด แยกตามคลาส"""
    idx = labels.long() * num_classes + preds.long()
    return torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)
 
 
def _safe_div(a, b):
    """หารอย่างปลอดภัย ป้องกันหารด้วยศูนย์"""
    return torch.where(b > 0, a / b.clamp_min(1e-12), torch.zeros_like(a))
 
 
def metrics_from_cm(cm):
    """คำนวณ Accuracy, Precision, Recall, F1 แบบ Macro"""
    cm = cm.double()
    tp = cm.diag()
    prec = _safe_div(tp, cm.sum(0))
    rec = _safe_div(tp, cm.sum(1))
    f1 = _safe_div(2 * prec * rec, prec + rec)
    return {
        'acc': (tp.sum() / cm.sum().clamp_min(1)).item(),
        'prec': prec.mean().item(),
        'recall': rec.mean().item(),
        'f1': f1.mean().item(),
    }
 
 
def binary_metrics(pred_pos, true_pos):
    """วัดผลแบบ 2 คลาส โดยถือ Perfect เป็น Positive"""
    tp = int((pred_pos & true_pos).sum())
    fp = int((pred_pos & ~true_pos).sum())
    fn = int((~pred_pos & true_pos).sum())
    tn = int((~pred_pos & ~true_pos).sum())
    total = tp + fp + fn + tn
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    acc = (tp + tn) / total if total else 0.0
    cm = torch.tensor([[tp, fn], [fp, tn]], dtype=torch.long)
    return {'acc': acc, 'prec': prec, 'recall': rec, 'f1': f1, 'cm': cm}
 
 
def make_threshold_grid(spec=(0.01, 0.99, 0.01), device='cpu'):
    """สร้างค่า Threshold ตั้งแต่ 0.01 ถึง 0.99 เพิ่มทีละ 0.01"""
    start, stop, step = spec
    return torch.arange(start, stop + step / 2, step, device=device, dtype=torch.float32)
 
 
def find_best_threshold(p_pos, true_pos, grid):
    """ทดลองทุก Threshold แล้วเลือกค่าที่ให้ F1 ของ Perfect สูงสุด"""
    grid = grid.to(p_pos.device)
    preds = p_pos.unsqueeze(0) >= grid.unsqueeze(1)  # (T, N)
    y = true_pos.unsqueeze(0)
    tp = (preds & y).sum(1).double()
    fp = (preds & ~y).sum(1).double()
    fn = (~preds & y).sum(1).double()
    denom = 2 * tp + fp + fn
    f1 = torch.where(denom > 0, 2 * tp / denom.clamp_min(1), torch.zeros_like(denom))
    best = int(torch.argmax(f1))
    return float(grid[best]), float(f1[best])
 
 
def compute_metrics(probs, labels, num_classes, pos_idx=None, threshold=None, grid=None):
    """All metrics from softmax probabilities.
 
    Keys:
      loss, acc, prec, recall, f1, cm                    -> all classes (argmax, macro)
      acc_2class, prec_2class, recall_2class, f1_2class, cm_2class
                                                          -> Perfect vs rest, argmax
      threshold, acc_perfect_at_thr, prec_perfect_at_thr, recall_perfect_at_thr, f1_perfect_at_thr, cm_thr
                                                          -> Perfect vs rest, P(Perfect) >= threshold
    threshold=None -> search the best one (use on val). Pass the val threshold for test.
    """
    out = {}
    labels = labels.long()
    p_true = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    out['loss'] = (-torch.log(p_true.clamp_min(1e-12))).mean().item()  # = CrossEntropy
 
    preds = probs.argmax(1)
    cm = confusion_matrix(preds, labels, num_classes)
    out.update(metrics_from_cm(cm))
    out['cm'] = cm.cpu()
 
    if pos_idx is not None:
        true_pos = labels == pos_idx
        p_pos = probs[:, pos_idx]
 
        b = binary_metrics(preds == pos_idx, true_pos)
        out.update({f'{k}_2class': v for k, v in b.items() if k != 'cm'})
        out['cm_2class'] = b['cm']
 
        if threshold is None:
            if grid is None:
                grid = make_threshold_grid(device=probs.device)
            threshold, _ = find_best_threshold(p_pos, true_pos, grid)
        t = binary_metrics(p_pos >= threshold, true_pos)
        out.update({f'{k}_perfect_at_thr': v for k, v in t.items() if k != 'cm'})
        out['cm_thr'] = t['cm']
        out['threshold'] = float(threshold)
 
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


def save_predictions(probs, labels, samples, class_names, out_path, pos_idx=None, threshold=None):
    """เขียน CSV 1 แถวต่อ 1 ภาพ (เรียงตาม dataset)
    samples = ImageFolder.samples -> [(path, class_idx), ...]
    outcome (Perfect = positive, ตัดสินด้วย P(Perfect) >= threshold):
      TP = Perfect -> Perfect | FP = Other -> Perfect | FN = Perfect -> Other | TN = Other -> Other
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
    has_bin = pos_idx is not None and threshold is not None
    if has_bin:
        pos_name, neg_name = class_names[pos_idx], f'non-{class_names[pos_idx]}'
        pred_pos = probs[:, pos_idx] >= threshold
        true_pos = labels == pos_idx

    fields = ['index', 'path', 'filename', 'true_label', 'pred_label', 'correct', 'confidence']
    fields += [f'p_{name}' for name in class_names]
    if has_bin:
        fields += ['threshold', 'true_binary', 'pred_at_thr', 'outcome']

    counts = {'n': n, 'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0} if has_bin else {'n': n}
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
                   'confidence': round(float(probs[i].max()), 5)}
            for j, name in enumerate(class_names):
                row[f'p_{name}'] = round(float(probs[i, j]), 5)
            if has_bin:
                tp_, pp_ = bool(true_pos[i]), bool(pred_pos[i])
                outcome = ('TP' if pp_ else 'FN') if tp_ else ('FP' if pp_ else 'TN')
                counts[outcome] += 1
                row.update({'threshold': round(float(threshold), 4),
                            'true_binary': pos_name if tp_ else neg_name,
                            'pred_at_thr': pos_name if pp_ else neg_name,
                            'outcome': outcome})
            writer.writerow(row)
    os.replace(tmp_path, out_path)   # atomic
    return counts