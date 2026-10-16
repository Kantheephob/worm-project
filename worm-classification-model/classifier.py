import os
import sys
import atexit
import traceback
import tempfile
import uuid
import warnings
from time import perf_counter
from pathlib import Path
from functools import partial
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
try:
    from torch.distributed.fsdp import fully_shard, CPUOffloadPolicy
except ImportError:
    from torch.distributed._composable.fsdp import fully_shard, CPUOffloadPolicy
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict, set_model_state_dict,
    get_state_dict, set_state_dict,
)
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from data import get_dataloaders, worm_transforms
from models import build_model, resolve_model_names, get_model_family, count_params
from logger import TrainLogger
from evaluate import (find_class_index, gather_eval_tensor, compute_metrics,
                      make_threshold_grid, benchmark_inference, save_predictions)
from common import (set_seed, find_free_port, safe_torch_save, atomic_json_dump,
                    save_dcp_checkpoint, load_dcp_checkpoint, dcp_model_to_pt,
                    dist_barrier, dist_all_ok, dist_broadcast_bool)
import config as c

# known-harmless warnings that flood the log every epoch
warnings.filterwarnings('ignore', message='Detected an existing checkpoint')              # DCP overwrite (intended)
warnings.filterwarnings('ignore', message=r'The epoch parameter in `scheduler\.step\(\)`')  # raised inside SequentialLR

if os.environ.get("LOCAL_RANK", "0") != "0" or os.environ.get("RANK", "0") != "0":
    sys.stdout = open(os.devnull, 'w')

LR_SCHEDULER_NAME = 'LinearLR(warmup) + CosineAnnealingLR'


# ============================================================
# Helpers
# ============================================================
def _is_cuda_fatal(exc):
    msg = str(exc)
    fatal_markers = ('CUDA error', 'illegal memory access', 'initialization error',
                     'killed by signal', 'CUDA out of memory', 'NCCL error')
    return any(marker in msg for marker in fatal_markers)


def _destroy_process_group_once():
    if dist.is_initialized():
        dist.destroy_process_group()


def _maybe_init_distributed():
    is_torchrun = "LOCAL_RANK" in os.environ
    if not is_torchrun:
        return False

    if not dist.is_initialized():
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("cuda:nccl,cpu:gloo")
        atexit.register(_destroy_process_group_once)

    return True


def _resolve_amp(a, device, is_main):
    """Returns (amp_enabled, amp_dtype, use_grad_scaler)."""
    if not a['use_amp'] or device.type != 'cuda':
        return False, torch.float32, False
    if a['amp_dtype'] == 'bf16':
        if torch.cuda.is_bf16_supported():
            return True, torch.bfloat16, False   # bf16 has fp32 range -> no GradScaler needed
        if is_main:
            print("⚠️ bf16 not supported on this GPU -> falling back to fp16 + GradScaler")
    return True, torch.float16, True


def _build_scheduler(optim, a, warmup_epochs):
    if warmup_epochs <= 0:
        return CosineAnnealingLR(optim, T_max=max(a['epochs'], 1), eta_min=a['eta_min'])
    warmup = LinearLR(optim, start_factor=a['start_factor'], end_factor=a['end_factor'],
                      total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optim, T_max=max(a['epochs'] - warmup_epochs, 1), eta_min=a['eta_min'])
    return SequentialLR(optim, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def _prefixed(metrics, prefix):
    """{'f1': .9, 'cm': ...} -> {'val_f1': .9}  (confusion matrices dropped)."""
    return {f'{prefix}_{k}': v for k, v in metrics.items() if not k.startswith('cm')}


def _offload_mode(value):
    """config / CLI value -> 'auto' | True | False"""
    if isinstance(value, str):
        v = value.strip().lower()
        if v == 'auto':
            return 'auto'
        return v in ('true', 'on', '1', 'yes')
    return bool(value)


DEFAULT_ARGS = {
    'img_size': c.IMG_SIZE,
    'upscale': c.UPSCALE,
    'augment': c.AUGMENT,
    'num_workers': c.NUM_WORKERS,
    'pin_memory': c.PIN_MEMORY,
    'epochs': c.EPOCHS,
    'warmup_epochs': c.WARMUP_EPOCHS,
    'batch_size': c.BATCH_SIZE,
    'lr': c.LR,
    'weight_decay': c.WEIGHT_DECAY,
    'start_factor': c.START_FACTOR,
    'end_factor': c.END_FACTOR,
    'eta_min': c.ETA_MIN,
    'use_amp': c.USE_AMP,
    'amp_dtype': c.AMP_DTYPE,
    'grad_clip_norm': c.GRAD_CLIP_NORM,
    'early_stop_patience': c.EARLY_STOP_PATIENCE,
    'fsdp_cpu_offload': c.FSDP_CPU_OFFLOAD,
    'fsdp_offload_mem_fraction': c.FSDP_OFFLOAD_MEM_FRACTION,
    'deterministic': c.DETERMINISTIC,
    'allow_tf32': c.ALLOW_TF32,
    'selection_metric': c.SELECTION_METRICS,
    'infer_batch_size': c.INFER_BATCH_SIZE,
    'infer_warmup': c.INFER_WARMUP,
    'infer_iters': c.INFER_ITERS,
    'save_val_predictions': c.SAVE_VAL_PREDICTIONS,
    'seed': c.SEED,
    'experiment_name': c.EXPERIMENT_NAME,
    'device': c.DEVICE,
    'output_dir': c.OUTPUT_DIR,
}


class WormClassificationModel:
    def __init__(self, model_name, experiment_name=None, **overrides):
        self.model_list = resolve_model_names(model_name)   # family name -> every variant of that family
        self.family = model_name.lower()
        self.init_overrides = dict(overrides)
        self.args = {**DEFAULT_ARGS, **overrides}
        self.experiment_name = experiment_name or self.args['experiment_name']

    # ============================================================
    # Public entry
    # ============================================================
    def train(self, dataset_dir, dataset_name, devices=None, **overrides):
        self.dataset_name = dataset_name

        temp_filename = f"worm_tmp_{self.experiment_name}_{self.family}_{uuid.uuid4().hex[:8]}.pt"
        temp_result_path = Path(tempfile.gettempdir()) / temp_filename

        is_torchrun = _maybe_init_distributed()
        results = {}

        try:
            if is_torchrun:
                rank = int(os.environ["RANK"])
                world_size = int(os.environ["WORLD_SIZE"])
                self._train_worker(rank, world_size, None, dataset_dir, dataset_name, temp_result_path,
                                   overrides, is_torchrun=True)
                if rank == 0 and temp_result_path.exists():
                    results = torch.load(temp_result_path, weights_only=False)
            else:
                if devices is None:
                    devices = [0] if torch.cuda.is_available() else ['cpu']
                elif isinstance(devices, int):
                    devices = [devices]

                world_size = len(devices)

                if world_size > 1 and devices[0] != 'cpu':
                    os.environ['MASTER_ADDR'] = '127.0.0.1'
                    os.environ['MASTER_PORT'] = find_free_port()
                    mp.spawn(self._train_worker, nprocs=world_size,
                             args=(world_size, devices, dataset_dir, dataset_name, temp_result_path,
                                   overrides, False),
                             join=True)
                else:
                    self._train_worker(0, 1, devices, dataset_dir, dataset_name, temp_result_path,
                                       overrides, False)

                if temp_result_path.exists():
                    results = torch.load(temp_result_path, weights_only=False)

        finally:
            if temp_result_path.exists():
                try:
                    temp_result_path.unlink()
                except Exception:
                    pass

        return results

    @staticmethod
    def benchmark_checkpoint(model_name, num_classes, pt_path, img_size=c.IMG_SIZE, device=c.DEVICE,
                             use_amp=c.USE_AMP, batch_size=c.INFER_BATCH_SIZE):
        """Measure latency / fps of any saved best.pt (standalone use)."""
        device = torch.device(device)
        a = {'use_amp': use_amp, 'amp_dtype': c.AMP_DTYPE}
        amp_enabled, amp_dtype, _ = _resolve_amp(a, device, True)
        model = build_model(model_name, num_classes, finetune_path=pt_path, device=device)
        return benchmark_inference(model, img_size, device, amp_enabled, amp_dtype, batch_size=batch_size,
                                   warmup=c.INFER_WARMUP, iters=c.INFER_ITERS)

    # ============================================================
    # One process (one GPU)
    # ============================================================
    def _train_worker(self, rank, world_size, devices, dataset_dir, dataset_name, temp_result_path,
                      overrides, is_torchrun):
        if rank != 0:
            sys.stdout = open(os.devnull, 'w')

        is_distributed = world_size > 1
        is_main = rank == 0
        args = {**self.args, **overrides}
        self.user_lr_set = 'lr' in self.init_overrides or 'lr' in overrides
        self.world_size = world_size

        if is_distributed:
            if is_torchrun:
                device_id = int(os.environ["LOCAL_RANK"])
            else:
                device_id = devices[rank]
                torch.cuda.set_device(device_id)
                dist.init_process_group("cuda:nccl,cpu:gloo", rank=rank, world_size=world_size)
            device = torch.device(f'cuda:{device_id}')
        else:
            device = torch.device(f'cuda:{devices[0]}' if devices[0] != 'cpu' else 'cpu')

        set_seed(args['seed'], deterministic=args['deterministic'])
        if args['allow_tf32']:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('high')

        base_dir = Path(args['output_dir']) / self.experiment_name
        ckpt_dir = base_dir / 'ckpt' / dataset_name / self.family
        log_dir = base_dir / 'log' / dataset_name / self.family
        tb_dir = base_dir / 'tb' / dataset_name / self.family
        pred_dir = base_dir / 'pred' / dataset_name / self.family

        if is_main:
            print(f"[WormClassificationModel] experiment = '{self.experiment_name}' | dataset = '{dataset_name}'")
            print(f"[WormClassificationModel] variants = {self.model_list}")
            print(f"[WormClassificationModel] Using {world_size} GPU(s) | FSDP2: {is_distributed} "
                  f"(CPU offload mode: {_offload_mode(args['fsdp_cpu_offload']) if is_distributed else 'n/a'}) "
                  f"| Torchrun: {is_torchrun}")
            print(f"[WormClassificationModel] deterministic={args['deterministic']} "
                  f"(cudnn.benchmark={not args['deterministic']}) | tf32={args['allow_tf32']}")
            print(f"[WormClassificationModel] img_size={args['img_size']} upscale={args['upscale']} "
                  f"augment={args['augment']} | batch/GPU={args['batch_size']} "
                  f"(effective {args['batch_size'] * world_size}) | patience={args['early_stop_patience']}")
            for d in (ckpt_dir, log_dir, tb_dir):
                d.mkdir(parents=True, exist_ok=True)

        warmup_epochs = min(args['warmup_epochs'], max(args['epochs'] - 1, 0))
        train_tf = partial(worm_transforms, img_size=args['img_size'], upscale=args['upscale'],
                           augment=args['augment'])
        eval_tf = partial(worm_transforms, img_size=args['img_size'], upscale=args['upscale'], augment=False)

        try:
            train_loader, val_loader, test_loader, class_names, num_classes, train_sampler = get_dataloaders(
                dataset_dir, train_tf, eval_tf, args['batch_size'], args['num_workers'], args['pin_memory'],
                is_distributed=is_distributed, is_main=is_main, seed=args['seed'],
            )
            self.class_names = class_names
            self.num_classes = num_classes
            self.pos_idx = find_class_index(class_names, c.POSITIVE_CLASS_NAMES)
            if is_main:
                if self.pos_idx is not None:
                    print(f"[Perfect] positive class = '{class_names[self.pos_idx]}' (index {self.pos_idx}) | "
                          f"best epoch selected by '{args['selection_metric']}'")
                else:
                    print(f"⚠️ [Perfect] none of POSITIVE_CLASS_NAMES={c.POSITIVE_CLASS_NAMES} found in "
                          f"{class_names} — Perfect/threshold metrics skipped, selecting by 'val_f1'")
            dataloaders_ok = True
        except Exception as e:
            print(f"❌ [rank {rank}] Failed to build dataloaders for '{dataset_name}': {e}", file=sys.stderr)
            traceback.print_exc()
            train_loader = val_loader = test_loader = train_sampler = None
            class_names, num_classes, dataloaders_ok = [], 0, False

        dataloaders_ok = dist_all_ok(dataloaders_ok)

        results = {}
        if not dataloaders_ok:
            for model_name in self.model_list:
                results[model_name] = {'error': True, 'ckpt_path': None}
        else:
            for model_name in self.model_list:
                if is_main:
                    print(f'\n{"=" * 60}\nTraining {model_name}\n{"=" * 60}')
                results[model_name] = self._train_one(
                    model_name, train_loader, val_loader, test_loader, train_sampler, args, warmup_epochs,
                    device, ckpt_dir, log_dir, tb_dir, pred_dir, is_main, is_distributed, rank,
                )

        if is_main:
            if not safe_torch_save(
                {**results, 'family': self.family, 'experiment_name': self.experiment_name,
                 'dataset_name': dataset_name, 'ckpt_dir': str(ckpt_dir), 'class_names': class_names},
                temp_result_path,
            ):
                print("⚠️ Failed to persist run results after retries; continuing.")

        if is_distributed:
            dist.barrier()
            if not is_torchrun:
                dist.destroy_process_group()

    # ============================================================
    # Model helpers
    # ============================================================
    def _variant_args(self, model_name, args):
        a = dict(args)
        fam = get_model_family(model_name)
        if not self.user_lr_set and fam in c.FAMILY_LR:
            a['lr'] = c.FAMILY_LR[fam]
        return a

    def _want_cpu_offload(self, param_count, device, a):
        """Decide FSDP CPU offload for this model.
        'auto': offload only if sharded fp32 params + grads + AdamW states (16 B/param / world)
                plus one unsharded fp32 copy during forward/backward (4 B/param)
                exceed fsdp_offload_mem_fraction of this GPU's memory.
        Same GPU model on every rank -> every rank makes the same decision."""
        mode = _offload_mode(a['fsdp_cpu_offload'])
        if mode != 'auto':
            return mode
        if device.type != 'cuda':
            return False
        total = torch.cuda.get_device_properties(device).total_memory
        world = max(self.world_size, 1)
        need = param_count * (16 / world + 4)
        return need > a['fsdp_offload_mem_fraction'] * total

    def _make_model(self, model_name, device, a, is_distributed, is_main=False, verbose=False):
        model = build_model(model_name, self.num_classes, device=device)
        model.to(device)
        param_count = count_params(model)
        if is_distributed:
            offload = self._want_cpu_offload(param_count, device, a)
            if verbose and is_main:
                print(f"[FSDP2] {model_name}: {param_count / 1e6:.1f}M params | CPU offload = {offload}")
            if offload:
                model = fully_shard(model, offload_policy=CPUOffloadPolicy())
            else:
                model = fully_shard(model)
        return model, param_count

    def _predict(self, model, loader, device, amp_enabled, amp_dtype):
        """Softmax probabilities + labels for the whole loader (gathered from all GPUs, dataset order)."""
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    outputs = model(imgs)
                all_probs.append(torch.softmax(outputs.float(), dim=1))
                all_labels.append(labels.long())

        if all_probs:
            probs, labels = torch.cat(all_probs), torch.cat(all_labels)
        else:
            probs = torch.zeros((0, self.num_classes), device=device)
            labels = torch.zeros((0,), dtype=torch.long, device=device)

        n = len(loader.dataset)
        return gather_eval_tensor(probs, n), gather_eval_tensor(labels, n)

    # ============================================================
    # One variant: train -> test -> benchmark -> log
    # ============================================================
    def _train_one(self, model_name, train_loader, val_loader, test_loader, train_sampler, args,
                   warmup_epochs, device, ckpt_dir, log_dir, tb_dir, pred_dir, is_main, is_distributed, rank=0):
        a = self._variant_args(model_name, args)
        selection = a['selection_metric'] if self.pos_idx is not None else 'val_f1'
        amp_enabled, amp_dtype, use_scaler = _resolve_amp(a, device, is_main)
        pos_name = self.class_names[self.pos_idx] if self.pos_idx is not None else None
        bin_names = [pos_name, f'non-{pos_name}'] if pos_name else None

        variant_ckpt_dir = ckpt_dir / model_name
        paths = {
            'best': variant_ckpt_dir / 'best',                # DCP dir, model only
            'last': variant_ckpt_dir / 'last',                # DCP dir, model + optim (resume)
            'last_meta': variant_ckpt_dir / 'last_meta.pt',   # scheduler / counters (resume)
            'best_meta': variant_ckpt_dir / 'best_meta.json', # best epoch + Perfect threshold
            'best_pt': variant_ckpt_dir / 'best.pt',          # plain state_dict
            'config': log_dir / f'{model_name}_config.json',
            'cm_val_best': log_dir / f'{model_name}_cm_val_best.png',
            'cm_val_best_thr': log_dir / f'{model_name}_cm_val_best_perfect_at_thr.png',
            'cm_test': log_dir / f'{model_name}_cm_test.png',
            'cm_test_thr': log_dir / f'{model_name}_cm_test_perfect_at_thr.png',
            'pred_val': pred_dir / model_name / 'val_predictions.csv',
            'pred_test': pred_dir / model_name / 'test_predictions.csv',
        }

        if is_main:
            variant_ckpt_dir.mkdir(parents=True, exist_ok=True)
            atomic_json_dump({
                'model_name': model_name,
                'family': get_model_family(model_name),
                'experiment_name': self.experiment_name,
                'dataset_name': self.dataset_name,
                'class_names': self.class_names,
                'positive_class': pos_name,
                'selection_metric': selection,
                'num_gpus': self.world_size,
                'effective_batch_size': a['batch_size'] * self.world_size,
                'optimizer': c.OPTIMIZER_NAME,
                'lr_scheduler': LR_SCHEDULER_NAME,
                'warmup_epochs_effective': warmup_epochs,
                'amp_dtype_effective': str(amp_dtype) if amp_enabled else 'fp32',
                'threshold_grid': c.THRESHOLD_GRID,
                'args': a,
            }, paths['config'])
        dist_barrier()

        out = None
        local_ok = True
        try:
            out = self._run_training_loop(model_name, train_loader, val_loader, train_sampler, a, warmup_epochs,
                                          device, log_dir, tb_dir, is_main, is_distributed, paths, selection,
                                          amp_enabled, amp_dtype, use_scaler, bin_names)
        except Exception as e:
            print(f"❌ [rank {rank}] Error training {model_name} on {self.dataset_name}: {e}", file=sys.stderr)
            traceback.print_exc()
            local_ok = False
            if _is_cuda_fatal(e):
                print(f"💀 [rank {rank}] CUDA context likely poisoned "
                      f"(fatal error while training {model_name}) — aborting job now.", file=sys.stderr)
                sys.stderr.flush()
                os._exit(1)

        if device.type == 'cuda':
            torch.cuda.empty_cache()

        all_ok = dist_all_ok(local_ok)
        logger = out['logger'] if out else None
        snap = out['best_snapshot'] if out else None

        if not all_ok or snap is None or not paths['best'].exists():
            if is_main:
                if all_ok:
                    print(f"⚠️ {model_name}: no checkpoint was ever saved — skipping test phase.")
                if logger is not None:
                    logger.close()
            return {'error': not all_ok, 'ckpt_path': str(paths['best_pt']) if paths['best_pt'].exists() else None}

        dist_barrier()

        # ---------------- test + per-image predictions (all ranks) ----------------
        test_m = None
        thr = snap.get('val_threshold')
        split_loaders = []
        if a['save_val_predictions']:
            split_loaders.append(('val', val_loader))
        if test_loader is not None:
            split_loaders.append(('test', test_loader))

        if split_loaders:
            model, _ = self._make_model(model_name, device, a, is_distributed, is_main=is_main, verbose=False)
            model_sd = get_model_state_dict(model)
            load_dcp_checkpoint({'model': model_sd}, paths['best'])
            set_model_state_dict(model, model_sd)

            for split, loader in split_loaders:
                probs, labels = self._predict(model, loader, device, amp_enabled, amp_dtype)
                if split == 'test':
                    test_m = compute_metrics(probs, labels, self.num_classes, self.pos_idx, threshold=thr)
                if is_main:
                    try:
                        counts = save_predictions(probs, labels, loader.dataset.samples, self.class_names,
                                                  paths[f'pred_{split}'], pos_idx=self.pos_idx, threshold=thr)
                        print(f"📝 {model_name}: saved {split} predictions -> {paths[f'pred_{split}']} {counts}")
                    except Exception as e:
                        print(f"⚠️ {model_name}: could not save {split} predictions: {e}")
                del probs, labels

            del model, model_sd
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        dist_barrier()

        # ---------------- inference benchmark (main GPU only) ----------------
        infer = {}
        if is_main and paths['best_pt'].exists():
            try:
                plain = build_model(model_name, self.num_classes, finetune_path=paths['best_pt'], device=device)
                infer = benchmark_inference(plain, a['img_size'], device, amp_enabled, amp_dtype,
                                            batch_size=a['infer_batch_size'], warmup=a['infer_warmup'],
                                            iters=a['infer_iters'])
                print(f"⚡ {model_name}: latency {infer['inference_latency_ms']:.2f} ms/img (bs=1) | "
                      f"{infer['inference_fps']:.0f} img/s (bs={a['infer_batch_size']})")
                del plain
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"⚠️ {model_name}: inference benchmark failed: {e}")
        dist_barrier()

        # ---------------- logging (main only) ----------------
        model_size_mb = paths['best_pt'].stat().st_size / 1024 ** 2 if paths['best_pt'].exists() else None
        test_p = _prefixed(test_m, 'test') if test_m is not None else {}

        if is_main:
            if test_m is not None:
                logger.log_confusion_matrix(test_m['cm'], self.class_names, snap['epoch'],
                                            'Confusion_Matrix/Test', png_path=paths['cm_test'])
                if 'cm_thr' in test_m:
                    logger.log_confusion_matrix(
                        test_m['cm_thr'], bin_names, snap['epoch'],
                        f"Confusion_Matrix/Test_{pos_name}_at_thr_{snap['val_threshold']:.2f}",
                        png_path=paths['cm_test_thr'])

            summary = {
                'variant': model_name,
                'dataset_name': self.dataset_name,
                'img_size': a['img_size'],
                'effective_batch_size': a['batch_size'] * self.world_size,
                'num_gpus': self.world_size,
                'optimizer': c.OPTIMIZER_NAME,
                'lr_scheduler': LR_SCHEDULER_NAME,
                'base_lr': a['lr'],
                'min_lr': a['eta_min'],
                'weight_decay': a['weight_decay'],
                'warmup_epochs': warmup_epochs,
                'warmup_ratio': warmup_epochs / a['epochs'] if a['epochs'] else 0.0,
                'warmup_start_factor': a['start_factor'],
                'warmup_end_factor': a['end_factor'],
                'max_epochs': a['epochs'],
                'early_stop_patience': a['early_stop_patience'],
                'grad_clip_norm': a['grad_clip_norm'],
                'param_count_m': out['param_count'] / 1e6,
                'best_epoch': snap['epoch'],
                'stopped_epoch': out['stopped_epoch'],
                'best_lr': snap['lr'],
                'total_train_time_sec': out['train_time_sec'],
                'seed': a['seed'],
                'use_amp': amp_enabled,
                'amp_dtype': ('bf16' if amp_dtype == torch.bfloat16 else 'fp16') if amp_enabled else 'fp32',
                'augment': a['augment'],
                'model_size_mb': model_size_mb,
                'inference_latency_ms': infer.get('inference_latency_ms'),
                'inference_fps': infer.get('inference_fps'),
                'optuna_tuned': False,
                'optuna_trial': None,
                'train_loss': snap['train_loss'],
                'train_acc': snap['train_acc'],
                'perfect_threshold': snap.get('val_threshold'),
                **{k: v for k, v in snap.items() if k.startswith('val_')},
                **test_p,
            }
            logger.log_summary(summary)

            hparam_metrics = {
                f'hp/best_{selection}': snap.get(selection),
                'hp/test_f1': test_p.get('test_f1'),
                'hp/test_f1_2class': test_p.get('test_f1_2class'),
                'hp/test_f1_perfect_at_thr': test_p.get('test_f1_perfect_at_thr'),
                'hp/inference_latency_ms': infer.get('inference_latency_ms'),
            }
            logger.log_hparams(
                hparams={'family': get_model_family(model_name), 'variant': model_name,
                         'experiment_name': self.experiment_name, 'img_size': a['img_size'],
                         'batch_size': a['batch_size'], 'lr': a['lr'], 'weight_decay': a['weight_decay'],
                         'warmup_epochs': warmup_epochs, 'eta_min': a['eta_min'], 'epochs': a['epochs'],
                         'seed': a['seed'], 'augment': a['augment']},
                metrics={k: v for k, v in hparam_metrics.items() if v is not None},
            )
            logger.close()

            if test_m is not None:
                msg = f"🧪 {model_name} TEST | macro F1 {test_m['f1']:.4f}"
                if 'f1_2class' in test_m:
                    msg += (f" | {pos_name} F1 {test_m['f1_2class']:.4f} (argmax)"
                            f" | {test_m['f1_perfect_at_thr']:.4f} @thr={snap['val_threshold']:.2f}")
                print(msg)

        return {
            'error': False,
            'best_epoch': snap['epoch'],
            'stopped_epoch': out['stopped_epoch'],
            'selection_metric': selection,
            'best_selection_value': snap.get(selection),
            'perfect_threshold': snap.get('val_threshold'),
            'base_lr': a['lr'],
            'param_count_m': out['param_count'] / 1e6,
            'model_size_mb': model_size_mb,
            'inference_latency_ms': infer.get('inference_latency_ms'),
            'inference_fps': infer.get('inference_fps'),
            'test_f1': test_p.get('test_f1'),
            'test_acc': test_p.get('test_acc'),
            'test_f1_2class': test_p.get('test_f1_2class'),
            'test_f1_perfect_at_thr': test_p.get('test_f1_perfect_at_thr'),
            'ckpt_path': str(paths['best_pt']),
        }

    # ============================================================
    # Training loop (resumable)
    # ============================================================
    def _run_training_loop(self, model_name, train_loader, val_loader, train_sampler, a, warmup_epochs,
                           device, log_dir, tb_dir, is_main, is_distributed, paths, selection,
                           amp_enabled, amp_dtype, use_scaler, bin_names):
        model, param_count = self._make_model(model_name, device, a, is_distributed, is_main=is_main, verbose=True)

        criterion = CrossEntropyLoss()
        optim = AdamW(model.parameters(), lr=a['lr'], weight_decay=a['weight_decay'])
        sched = _build_scheduler(optim, a, warmup_epochs)
        scaler = GradScaler(device=device.type, enabled=use_scaler)

        logger = TrainLogger(model_name, log_dir=log_dir, tb_dir=tb_dir) if is_main else None
        grid = make_threshold_grid(c.THRESHOLD_GRID, device=device)
        metric_key = selection[len('val_'):]   # 'val_f1_perfect_at_thr' -> 'f1_perfect_at_thr'

        st = {'epoch': 0, 'best_value': float('-inf'), 'best_epoch': 0, 'patience': 0,
              'best_snapshot': None, 'train_time_sec': 0.0, 'finished': False, 'param_count': param_count}

        # ---------------- resume ----------------
        if paths['last'].exists() and any(paths['last'].iterdir()) and paths['last_meta'].exists():
            meta = torch.load(paths['last_meta'], weights_only=False)
            model_sd, optim_sd = get_state_dict(model, optim)
            ck = {'model': model_sd, 'optim': optim_sd}
            load_dcp_checkpoint(ck, paths['last'])
            set_state_dict(model, optim, model_state_dict=ck['model'], optim_state_dict=ck['optim'])
            sched.load_state_dict(meta['sched'])
            if use_scaler and meta.get('scaler'):
                scaler.load_state_dict(meta['scaler'])
            for k in st:
                if k in meta:
                    st[k] = meta[k]
            if is_main:
                state = 'already finished -> going to test' if st['finished'] else f"at epoch {st['epoch']}"
                print(f"🔄 Resuming {model_name} {state} "
                      f"(best {selection}={st['best_value']:.4f} @ epoch {st['best_epoch']})")
        was_finished = st['finished']

        def save_meta():
            if is_main:
                meta = {**st, 'sched': sched.state_dict(), 'scaler': scaler.state_dict() if use_scaler else None}
                safe_torch_save(meta, paths['last_meta'])
            dist_barrier()

        # ---------------- epochs ----------------
        for epoch in range(st['epoch'] + 1, a['epochs'] + 1):
            if st['finished']:
                break
            t0 = perf_counter()
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            lr_used = optim.param_groups[0]['lr']            # LR actually used in THIS epoch
            wd_used = optim.param_groups[0]['weight_decay']

            # ---- train ----
            model.train()
            loss_sum = torch.zeros((), device=device)
            correct = torch.zeros((), device=device)
            seen = torch.zeros((), device=device)
            for imgs, labels in train_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optim.zero_grad(set_to_none=True)
                with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    outputs = model(imgs)
                    loss = criterion(outputs.float(), labels)
                scaler.scale(loss).backward()
                if a['grad_clip_norm'] > 0:
                    scaler.unscale_(optim)
                    clip_grad_norm_(model.parameters(), max_norm=a['grad_clip_norm'])
                scaler.step(optim)
                scaler.update()

                bs = labels.size(0)
                loss_sum += loss.detach().float() * bs
                correct += (outputs.detach().argmax(dim=1) == labels).sum()
                seen += bs

            totals = torch.stack([loss_sum, correct, seen])
            if is_distributed:
                dist.all_reduce(totals)
            train_loss = (totals[0] / totals[2].clamp_min(1)).item()
            train_acc = (totals[1] / totals[2].clamp_min(1)).item()
            sched.step()

            # ---- validate (threshold searched here) ----
            probs, labels = self._predict(model, val_loader, device, amp_enabled, amp_dtype)
            m = compute_metrics(probs, labels, self.num_classes, self.pos_idx, threshold=None, grid=grid)
            del probs, labels
            value = m[metric_key]
            improved = dist_broadcast_bool(value > st['best_value'])
            epoch_time = perf_counter() - t0

            if is_main:
                row = {
                    'epoch': epoch, 'current_lr': lr_used, 'current_weight_decay': wd_used,
                    'train_loss': train_loss, 'train_acc': train_acc,
                    'val_loss': m['loss'], 'val_acc': m['acc'], 'val_prec': m['prec'],
                    'val_recall': m['recall'], 'val_f1': m['f1'],
                    'val_f1_perfect': m.get('f1_2class'),
                    'val_best_threshold': m.get('threshold'),
                    'val_f1_perfect_at_thr': m.get('f1_perfect_at_thr'),
                    'epoch_time_sec': epoch_time,
                }
                logger.log_epoch(row)
                logger.log_confusion_matrix(m['cm'], self.class_names, epoch, 'Confusion_Matrix/Validation')

                msg = (f" Ep {epoch:>3}/{a['epochs']} | {model_name} | loss {train_loss:.4f}/{m['loss']:.4f}"
                       f" | val macro F1 {m['f1']:.4f}")
                if 'f1_2class' in m:
                    msg += f" | Perfect F1 {m['f1_2class']:.4f} | @thr {m['f1_perfect_at_thr']:.4f} (t={m['threshold']:.2f})"
                msg += f" | LR {lr_used:.2e} | {epoch_time:.0f}s" + (" 🏆" if improved else "")
                print(msg)

            # ---- best ----
            if improved:
                st['best_value'] = value
                st['best_epoch'] = epoch
                st['patience'] = 0
                st['best_snapshot'] = {'epoch': epoch, 'lr': lr_used, 'train_loss': train_loss,
                                       'train_acc': train_acc, **_prefixed(m, 'val')}
                save_dcp_checkpoint({'model': get_model_state_dict(model)}, paths['best'])
                if is_main:
                    atomic_json_dump({
                        'model_name': model_name, 'dataset_name': self.dataset_name, 'epoch': epoch,
                        'selection_metric': selection, 'selection_value': value,
                        'threshold': m.get('threshold'),
                        'positive_class': self.class_names[self.pos_idx] if self.pos_idx is not None else None,
                        'positive_index': self.pos_idx, 'class_names': self.class_names,
                        'img_size': a['img_size'], 'upscale': a['upscale'],
                    }, paths['best_meta'])
                    logger.save_confusion_matrix_png(m['cm'], self.class_names,
                                                     f'{model_name} | Val (best epoch {epoch})',
                                                     paths['cm_val_best'])
                    if 'cm_thr' in m:
                        logger.save_confusion_matrix_png(
                            m['cm_thr'], bin_names,
                            f"{model_name} | Val {bin_names[0]} @thr={m['threshold']:.2f} (epoch {epoch})",
                            paths['cm_val_best_thr'])
            else:
                st['patience'] += 1

            # ---- last (resume) ----
            st['epoch'] = epoch
            st['train_time_sec'] += epoch_time
            model_sd, optim_sd = get_state_dict(model, optim)
            save_dcp_checkpoint({'model': model_sd, 'optim': optim_sd}, paths['last'])

            stop = st['patience'] >= a['early_stop_patience']
            if stop or epoch == a['epochs']:
                st['finished'] = True
            save_meta()

            if stop:
                if is_main:
                    print(f"🛑 {model_name} early stopping at epoch {epoch} "
                          f"(best {selection}={st['best_value']:.4f} @ epoch {st['best_epoch']})")
                break

        if not st['finished']:
            st['finished'] = True
            save_meta()

        del model, optim
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # ---------------- best DCP -> plain best.pt ----------------
        dist_barrier()
        if paths['best'].exists() and (not was_finished or not paths['best_pt'].exists()):
            plain = build_model(model_name, self.num_classes, device=torch.device('cpu'))
            if dcp_model_to_pt(plain, paths['best'], paths['best_pt'], is_main=is_main) and is_main:
                print(f"💾 Saved plain .pt for best {model_name} at {paths['best_pt']}")
            del plain
        dist_barrier()

        return {'best_snapshot': st['best_snapshot'], 'stopped_epoch': st['epoch'], 'logger': logger,
                'train_time_sec': st['train_time_sec'], 'param_count': st['param_count']}