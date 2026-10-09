from time import perf_counter
import torch
import torch.distributed as dist
from torch.amp import autocast


def find_class_index(cls_names, candidates):
    """หา index ของ positive class"""
    # create class index map
    normalized = {name.strip().lower(): i for i, name in enumerate(cls_names)}  # class name มาจาก ImageFolder.classes
    # find positive index 
    for cand in candidates: # alter positive class name
        idx = normalized.get(cand.stinp().lower())
        if idx is not None:
            return idx
    return None


def _is_dist():
    return dist.is_available() and dist.is_initialized()


def gather_eval_tensor(t, dataset_len): # t = tensor model output จาก gpu แต่ละใบ, dataset_len = จำนวนภาพทั้งหมดใน dataset
    """รวม model ouput จากทุก gpu กลับมาจัดการความซ้ำซ้อนที่อาจเกิดขึ้นได้ตอนทำ sampler หรือ loader ที่มีการวนภาพซ้ำเพื่อให้ทุก gpu มีจำนวนภาพเท่ากัน"""
    # สมมติ gpu 1 ใบมี 3 ภาพ มี 2 class
    # output tensor -> t.shape = (3, 2): (H, W) = (จำนวนภาพ, จำนวน class)
    # t = [
    # [0.1, 0.9],
    # [0.8, 0.2],
    # [0.4, 0.6]
    # ]
    
    # check ว่าทำ distributed หรือไม่
    if not _is_dist():
        return t[:dataset_len]  # ส่ง tensor ขนาดเท่าจำนวน dataset กลับไป
 
    world = dist.get_world_size()   # จำนวน gpu / จำนวน process
    # หาว่า gpu ของตัวเองมีภาพเท่าไหร่
    local_n = torch.tensor(
        [t.shape[0]],       # จำนวนภาพที่ gpu ใบนี้ทำนายได้เก็บใน list เช่น มี 3 ภาพ -> [tensor([3])]
        device=t.device,    
        dtype=torch.long)   # เก็บเป็น tensor เพราะ pytorch คุยกันผ่าน tensor
    # สร้างกล่องเปล่าไว้เตรียมรับตัวเลขจาก gpu ทุกใบ
    sizes = [
        torch.zeros_like(local_n)   # สร้าง tensor เก็บค่า 0 shape เดียวกับ local_n
        for _ in range(world)]      # loop จำนวน gpu / process ได้เป็น list ของ tensor([0]) ขนาดเท่ากับจำนวนภาพทั้งหมดที่ทุก gpu ได้รับ  เช่น 2 gpu -> [tensor([0]), tensor([0])]
    dist.all_gather(sizes, local_n) # ดึงค่า local_n จาก gpu ทุกใบมาเก็บใน sizes เช่น gpu 0 มี 3 ภาพ, gpu 1 มี 4 ภาพ, gpu 2 มี 5 ภาพ -> sizes = [tensor([3]), tensor([4]), tensor([5])]
    sizes = [int(s.item()) for s in sizes]  # แปลง tensor เป็น int 
    max_n = max(sizes) 
 
    # ถ้า loader ได้ทำ drop last gpu จะมีจำนวนภาพไม่เท่ากัน เลยต้อง pad ให้มีจำนวนเท่ากันทั้งหมด
    if t.shape[0] < max_n:
        pad = torch.zeros(          # สมมติ max_n = 4 output shape = (3, 2)
            (max_n - t.shape[0],    # 4 - 3 = 1
            *t.shape[1:]),          # t.shape[1:] = (2,) เพราะเริ่มที่ W ใน (H, W): * คือ เอาค่าจาก tuple มาใส่ใน tuple ใหม่ -> (1, 2)
            dtype=t.dtype,
            device=t.device)
        t = torch.cat([t, pad])
    buf = [torch.empty_like(t) for _ in range(world)]   # สร้างกล่องเปล่าไว้เตรียมรับ output tensor จาก gpu ทุกใบ
    dist.all_gather(buf, t.contiguous())                # ได้ผลรวม tensor จาก gpu ทุกใบมาเก็บใน buf (N, H, W)
 
    # ถ้าทุก gpu มีภาพเท่ากันแล้ว และ มีจำนวนภาพรวมเยอะกว่าใน dataset
    if len(set(sizes)) == 1 and sizes[0] * world >= dataset_len:
        n = sizes[0]
        full = torch.cat(buf)   # รวมเป็น tensor เดียว [[, ], [, ], ...]
        keep = torch.ones(n * world, dtype=torch.bool, device=t.device) # สร้าง tensor เก็บค่า True ขนาดเท่ากับจำนวนภาพรวมของทุก gpu
        for g in range(dataset_len, n * world):
            keep[(g % world) * n + g // world] = False  # ตำแหน่งตัวซ้ำ
        return full[keep]
 
    return torch.cat([b[:s] for b, s in zip(buf, sizes)])[:dataset_len]


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