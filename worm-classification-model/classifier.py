import os
import sys
import atexit
import traceback
import tempfile
import uuid
from time import perf_counter
from pathlib import Path
from functools import partial

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
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
from models import build_model, resolve_model_names
from logger import TrainLogger
from evaluate import build_metrics, find_good_class_index, build_label_collapse_map
from common import set_seed, find_free_port, safe_torch_save, save_dcp_checkpoint, load_dcp_checkpoint, dcp_model_to_pt
import config as c

if os.environ.get("LOCAL_RANK", "0") != "0" or os.environ.get("RANK", "0") != "0":
    sys.stdout = open(os.devnull, 'w')


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


DEFAULT_ARGS = {
    'img_size': c.IMG_SIZE,
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
    'grad_clip_norm': c.GRAD_CLIP_NORM,
    'early_stop_patience': c.EARLY_STOP_PATIENCE,
    'seed': c.SEED,
    'experiment_name': c.EXPERIMENT_NAME,
    'device': c.DEVICE,
    'output_dir': c.OUTPUT_DIR,
}


class WormClassificationModel:
    def __init__(self, model_name, experiment_name=None, **overrides):
        self.model_list = resolve_model_names(model_name)
        self.family = model_name.lower()
        self.args = {**DEFAULT_ARGS, **overrides}
        self.experiment_name = experiment_name or self.args['experiment_name']

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
                self._train_worker(rank, world_size, None, dataset_dir, dataset_name, temp_result_path, overrides, is_torchrun=True)
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
                             args=(world_size, devices, dataset_dir, dataset_name, temp_result_path, overrides, False),
                             join=True)
                else:
                    self._train_worker(0, 1, devices, dataset_dir, dataset_name, temp_result_path, overrides, False)

                if temp_result_path.exists():
                    results = torch.load(temp_result_path, weights_only=False)
                    
        finally:
            if temp_result_path.exists():
                try:
                    temp_result_path.unlink()
                except Exception:
                    pass

        return results

    def _train_worker(self, rank, world_size, devices, dataset_dir, dataset_name, temp_result_path, overrides, is_torchrun):
        is_distributed = world_size > 1
        is_main = rank == 0
        args = {**self.args, **overrides}

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

        set_seed(args['seed'])

        base_dir = Path(args['output_dir']) / self.experiment_name
        ckpt_dir = base_dir / 'ckpt' / dataset_name / self.family
        log_dir = base_dir / 'log' / dataset_name / self.family
        tb_dir = base_dir / 'tb' / dataset_name / self.family

        if is_main:
            print(f"[WormClassificationModel] experiment = '{self.experiment_name}' | dataset = '{dataset_name}'")
            print(f"[WormClassificationModel] Using {world_size} GPUs (FSDP2 + CPU Offload Mode: {is_distributed} | Torchrun: {is_torchrun})")
            for d in (ckpt_dir, log_dir, tb_dir):
                d.mkdir(parents=True, exist_ok=True)

        warmup_epochs = min(args['warmup_epochs'], max(args['epochs'] - 1, 1))
        transform = partial(worm_transforms, img_size=args['img_size'])

        try:
            train_loader, val_loader, test_loader, class_names, num_classes, train_sampler = get_dataloaders(
                dataset_dir, transform, args['batch_size'], args['num_workers'], args['pin_memory'],
                is_distributed=is_distributed, is_main=is_main
            )
            self.class_names = class_names
            self.good_idx = find_good_class_index(class_names, c.GOOD_CLASS_NAMES)
            if is_main and len(class_names) > 2:
                if self.good_idx is not None:
                    print(f"[2-class] ใช้ '{class_names[self.good_idx]}' (index {self.good_idx}) "
                        f"เป็น positive class สำหรับ 2-class metric ของ '{dataset_name}'")
                else:
                    print(f"⚠️ [2-class] '{dataset_name}' มี {len(class_names)} classes {class_names} "
                        f"แต่ไม่มีอันไหนตรงกับ GOOD_CLASS_NAMES={c.GOOD_CLASS_NAMES} ใน config.py "
                        f"— จะข้ามการคำนวณ 2-class metric สำหรับ dataset นี้")
            dataloaders_ok = True
        except Exception as e:
            print(f"❌ [rank {rank}] Failed to build dataloaders for '{dataset_name}': {e}", file=sys.stderr)
            traceback.print_exc()
            train_loader = val_loader = test_loader = train_sampler = None
            class_names, num_classes, dataloaders_ok = [], 0, False

        if is_distributed:
            ok_tensor = torch.tensor([1 if dataloaders_ok else 0], device=device)
            dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)
            dataloaders_ok = bool(ok_tensor.item())

        results = {}
        if not dataloaders_ok:
            for model_name in self.model_list:
                results[model_name] = {'best_f1': 0.0, 'best_epoch': 0, 'test_f1': None,
                                        'test_accuracy': None, 'ckpt_path': None, 'error': True}
        else:
            for model_name in self.model_list:
                if is_main:
                    print(f'\n{"=" * 60}\nTraining {model_name}\n{"=" * 60}')
                results[model_name] = self._train_one(
                    model_name, num_classes, train_loader, val_loader, test_loader, train_sampler,
                    args, warmup_epochs, device, ckpt_dir, log_dir, tb_dir, is_main, is_distributed, rank,
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

    def _train_one(self, model_name, num_classes, train_loader, val_loader, test_loader, train_sampler,
                    args, warmup_epochs, device, ckpt_dir, log_dir, tb_dir, is_main, is_distributed, rank=0):
        variant_ckpt_dir = ckpt_dir / model_name
        
        if is_main:
            variant_ckpt_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = variant_ckpt_dir / 'best'      # DCP dir, model-only
        latest_ckpt_path = variant_ckpt_dir / 'last'  # DCP dir, model+optim (resume)
        logger = None
        best_f1, best_epoch = 0.0, 0
        best_snapshot = None
        local_ok = True

        train_start_time = perf_counter()
        try:
            best_f1, best_epoch, best_snapshot, logger, avg_train_loss = self._run_training_loop(
                model_name, num_classes, train_loader, val_loader, train_sampler,
                args, warmup_epochs, device, log_dir, tb_dir, is_main, is_distributed,
                ckpt_path, latest_ckpt_path,
            )
        except Exception as e:
            print(f"❌ [rank {rank}] Error training {model_name} on {self.dataset_name}: {e}", file=sys.stderr)
            traceback.print_exc()
            local_ok = False
            if _is_cuda_fatal(e):
                print(f"💀 [rank {rank}] CUDA context likely poisoned "
                      f"(fatal error while training {model_name}) — aborting job now.", file=sys.stderr)
                sys.stderr.flush()
                os._exit(1)

        total_train_time = perf_counter() - train_start_time

        if is_distributed:
            ok_tensor = torch.tensor([1 if local_ok else 0], device=device)
            dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)
            all_ok = bool(ok_tensor.item())
        else:
            all_ok = local_ok

        if not all_ok or best_snapshot is None or not ckpt_path.exists():
            if is_main:
                if all_ok:
                    print(f"⚠️ {model_name}: no checkpoint was ever saved — skipping test phase.")
                if logger is not None:
                    logger.close()
            return {'best_f1': best_f1, 'best_epoch': best_epoch, 'test_f1': None,
                    'test_accuracy': None, 'ckpt_path': str(ckpt_path) if ckpt_path.exists() else None,
                    'error': not all_ok}

        if is_distributed:
            dist.barrier()

        model = build_model(model_name, num_classes, device=device)
        model.to(device)
        if is_distributed:
            model = fully_shard(model, offload_policy=CPUOffloadPolicy())

        model_sd = get_model_state_dict(model)
        load_dcp_checkpoint({'model': model_sd}, ckpt_path)
        set_model_state_dict(model, model_sd)

        model.eval()
        criterion = CrossEntropyLoss()
        
        # Initialize test metrics to None by default
        test_loss = test_acc = test_prec = test_rec = test_f1 = None
        test_acc_2c = test_prec_2c = test_rec_2c = test_f1_2c = None

        collapse_needed = len(self.class_names) > 2 and self.good_idx is not None

        # Only evaluate if test_loader exists
        if test_loader is not None:
            test_metrics = build_metrics(num_classes, device)
            running_test_loss = 0.0

            if collapse_needed:
                collapse_map = build_label_collapse_map(self.class_names, self.good_idx).to(device)
                collapsed_class_names = ['good', 'other']
                collapsed_metrics = build_metrics(2, device)

            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    with autocast(device_type=device.type, enabled=args['use_amp']):
                        outputs = model(imgs)
                        loss = criterion(outputs.float(), labels)
                    running_test_loss += loss.item() * imgs.shape[0]
                    preds = outputs.argmax(dim=1)
                    test_metrics.update(preds, labels)
                    if collapse_needed:
                        collapsed_metrics.update(collapse_map[preds], collapse_map[labels])

            test_result = test_metrics.compute()
            test_loss = running_test_loss / len(test_loader.dataset)
            test_acc = test_result['accuracy'].item()
            test_prec = test_result['precision'].item()
            test_rec = test_result['recall'].item()
            test_f1 = test_result['f1'].item()
            
            if collapse_needed:
                collapsed_result = collapsed_metrics.compute()
                test_acc_2c = collapsed_result['accuracy'].item()
                test_prec_2c = collapsed_result['precision'].item()
                test_rec_2c = collapsed_result['recall'].item()
                test_f1_2c = collapsed_result['f1'].item()

        if is_main:
            # Only log test confusion matrices if test set exists
            if test_loader is not None:
                logger.log_confusion_matrix(test_result['confusion_matrix'], self.class_names, best_epoch, 'Confusion_Matrix/Test')
                if collapse_needed:
                    logger.log_confusion_matrix(collapsed_result['confusion_matrix'], collapsed_class_names,
                                                best_epoch, 'Confusion_Matrix/Test_2class')

            summary_kwargs = {
                'test_accuracy_2class': test_acc_2c,
                'test_precision_2class': test_prec_2c,
                'test_recall_2class': test_rec_2c,
                'test_f1_2class': test_f1_2c,
            }
            if best_snapshot is not None:
                summary_kwargs.update({
                    'val_accuracy_2class': best_snapshot.get('val_accuracy_2class'),
                    'val_precision_2class': best_snapshot.get('val_precision_2class'),
                    'val_recall_2class': best_snapshot.get('val_recall_2class'),
                    'val_f1_2class': best_snapshot.get('val_f1_2class'),
                })

            hparam_metrics = {'best_val_f1': best_f1, 'best_epoch': best_epoch}
            if test_loader is not None:
                hparam_metrics['test_f1'] = test_f1
                hparam_metrics['test_accuracy'] = test_acc
                if collapse_needed:
                    hparam_metrics['test_f1_2class'] = test_f1_2c
                    hparam_metrics['test_accuracy_2class'] = test_acc_2c

            logger.log_summary(best_epoch, args['seed'], avg_train_loss, best_snapshot['val_loss'],
                               best_snapshot['val_accuracy'], best_snapshot['val_precision'],
                               best_snapshot['val_recall'], best_f1, test_loss, test_acc,
                               test_prec, test_rec, test_f1, total_train_time, **summary_kwargs)
            logger.log_hparams(
                hparams={'family': self.family, 'variant': model_name, 'experiment_name': self.experiment_name,
                         'img_size': args['img_size'], 'batch_size': args['batch_size'], 'lr': args['lr'],
                         'weight_decay': args['weight_decay'], 'epochs': args['epochs'], 'seed': args['seed']},
                metrics=hparam_metrics
            )
            logger.close()

        result = {'best_f1': best_f1, 'best_epoch': best_epoch, 'test_f1': test_f1,
                  'test_accuracy': test_acc, 'ckpt_path': str(ckpt_path)}
        if collapse_needed:
            result['test_f1_2class'] = test_f1_2c
            result['test_accuracy_2class'] = test_acc_2c
            
        return result

    def _run_training_loop(self, model_name, num_classes, train_loader, val_loader, train_sampler,
                            args, warmup_epochs, device, log_dir, tb_dir, is_main, is_distributed,
                            ckpt_path, latest_ckpt_path):
        model = build_model(model_name, num_classes, device=device)
        model.to(device)
        if is_distributed:
            model = fully_shard(model, offload_policy=CPUOffloadPolicy())

        criterion = CrossEntropyLoss()
        optim = AdamW(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

        warmup_sched = LinearLR(optim, start_factor=args['start_factor'], end_factor=args['end_factor'], total_iters=warmup_epochs)
        cosine_sched = CosineAnnealingLR(optim, T_max=max(args['epochs'] - warmup_epochs, 1), eta_min=args['eta_min'])
        sched = SequentialLR(optim, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])
        scaler = GradScaler(device=device.type, enabled=args['use_amp'])

        logger = TrainLogger(model_name, log_dir=log_dir, tb_dir=tb_dir) if is_main else None

        train_metrics = build_metrics(num_classes, device)
        val_metrics = build_metrics(num_classes, device)

        collapse_needed = len(self.class_names) > 2 and self.good_idx is not None
        if collapse_needed:
            collapse_map = build_label_collapse_map(self.class_names, self.good_idx).to(device)
            val_collapsed_metrics = build_metrics(2, device)
            
        best_f1, best_epoch, patience_counter = 0.0, 0, 0
        best_snapshot = None
        total_train_loss_sum = 0.0
        start_epoch = 1
        epoch = start_epoch - 1

        if latest_ckpt_path.exists() and any(latest_ckpt_path.iterdir()):
            model_sd, optim_sd = get_state_dict(model, optim)
            resume_state = {'model': model_sd, 'optim': optim_sd, 'sched': sched.state_dict(),
                             'scaler': scaler.state_dict(), 'epoch': 0, 'best_f1': 0.0, 'best_epoch': 0,
                             'patience_counter': 0, 'total_train_loss_sum': 0.0, 'best_snapshot': None}
            load_dcp_checkpoint(resume_state, latest_ckpt_path)
            set_state_dict(model, optim, model_state_dict=resume_state['model'], optim_state_dict=resume_state['optim'])
            sched.load_state_dict(resume_state['sched'])
            scaler.load_state_dict(resume_state['scaler'])

            best_f1 = resume_state['best_f1']
            best_epoch = resume_state['best_epoch']
            patience_counter = resume_state['patience_counter']
            total_train_loss_sum = resume_state['total_train_loss_sum']
            best_snapshot = resume_state['best_snapshot']
            start_epoch = resume_state['epoch'] + 1
            epoch = resume_state['epoch']

            if is_main:
                print(f"🔄 Resuming {model_name} from '{latest_ckpt_path}' at epoch {resume_state['epoch']} "
                      f"(best_f1={best_f1:.4f} @ epoch {best_epoch})")

        for epoch in range(start_epoch, args['epochs'] + 1):
            start_time = perf_counter()
            if is_distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            model.train()
            train_metrics.reset()
            running_loss = 0.0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optim.zero_grad(set_to_none=True)
                with autocast(device_type=device.type, enabled=args['use_amp']):
                    outputs = model(imgs)
                    loss = criterion(outputs.float(), labels)
                scaler.scale(loss).backward()
                if args['grad_clip_norm'] > 0:
                    scaler.unscale_(optim)
                    clip_grad_norm_(model.parameters(), max_norm=args['grad_clip_norm'])
                scaler.step(optim)
                scaler.update()
                running_loss += loss.item() * imgs.shape[0]
                train_metrics.update(outputs.argmax(dim=1), labels)

            sched.step()
            train_result = train_metrics.compute()
            train_loss = running_loss / len(train_loader.dataset)
            total_train_loss_sum += train_loss

            model.eval()
            val_metrics.reset()
            if collapse_needed:
                val_collapsed_metrics.reset()
                
            running_val_loss = 0.0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    with autocast(device_type=device.type, enabled=args['use_amp']):
                        outputs = model(imgs)
                        loss = criterion(outputs.float(), labels)
                    running_val_loss += loss.item() * imgs.shape[0]
                    
                    preds = outputs.argmax(dim=1)
                    val_metrics.update(preds, labels)
                    if collapse_needed:
                        val_collapsed_metrics.update(collapse_map[preds], collapse_map[labels])

            val_result = val_metrics.compute()
            if collapse_needed:
                val_collapsed_result = val_collapsed_metrics.compute()
                
            val_loss = running_val_loss / len(val_loader.dataset)
            current_f1 = val_result['f1'].item()

            if is_main:
                elapsed = perf_counter() - start_time
                current_lr = optim.param_groups[0]['lr']
                print(f" Ep: {epoch}/{args['epochs']} | {model_name} | Val F1: {current_f1:.4f} | LR: {current_lr:.2e} | Time: {elapsed:.1f}s")
                logger.log_epoch(epoch, current_lr, train_loss, train_result['accuracy'].item(), val_loss,
                                  val_result['accuracy'].item(), val_result['precision'].item(),
                                  val_result['recall'].item(), current_f1, elapsed)
                logger.log_confusion_matrix(val_result['confusion_matrix'], self.class_names, epoch, 'Confusion_Matrix/Validation')

            improved = current_f1 > best_f1
            if improved:
                best_f1 = current_f1
                best_epoch = epoch
                patience_counter = 0
                best_snapshot = {'train_loss': train_loss, 'val_loss': val_loss, 'val_accuracy': val_result['accuracy'].item(),
                                  'val_precision': val_result['precision'].item(), 'val_recall': val_result['recall'].item(), 'val_f1': current_f1}

                if collapse_needed:
                    best_snapshot.update({
                        'val_accuracy_2class': val_collapsed_result['accuracy'].item(),
                        'val_precision_2class': val_collapsed_result['precision'].item(),
                        'val_recall_2class': val_collapsed_result['recall'].item(),
                        'val_f1_2class': val_collapsed_result['f1'].item(),
                    })
            else:
                patience_counter += 1

            model_sd, optim_sd = get_state_dict(model, optim)
            latest_state = {'model': model_sd, 'optim': optim_sd, 'sched': sched.state_dict(), 'scaler': scaler.state_dict(),
                             'epoch': epoch, 'best_f1': best_f1, 'best_epoch': best_epoch,
                             'patience_counter': patience_counter, 'total_train_loss_sum': total_train_loss_sum,
                             'best_snapshot': best_snapshot}
            if save_dcp_checkpoint(latest_state, latest_ckpt_path) and is_main:
                print(f"💾 Saved latest {model_name} (epoch {epoch}) at {latest_ckpt_path}")

            if improved:
                model_sd = get_model_state_dict(model)
                if save_dcp_checkpoint({'model': model_sd}, ckpt_path) and is_main:
                    print(f"🏆 Save best {model_name} (best_f1={best_f1:.4f}) at {ckpt_path}")

            if patience_counter >= args['early_stop_patience']:
                if is_main:
                    print(f'🛑 {model_name} early stopping at epoch {epoch}')
                break

        if ckpt_path.exists():
            plain_model = build_model(model_name, num_classes, device=device)
            pt_path = ckpt_path.with_suffix('.pt')
            if dcp_model_to_pt(plain_model, ckpt_path, pt_path, is_main=is_main) and is_main:
                print(f"💾 Saved plain .pt for best {model_name} at {pt_path}")
            del plain_model
            
        avg_train_loss = total_train_loss_sum / epoch if epoch > 0 else 0.0
        return best_f1, best_epoch, best_snapshot, logger, avg_train_loss