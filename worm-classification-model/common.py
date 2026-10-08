import csv
import os
import random
import socket
import shutil
import tempfile
import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict


def set_seed(seed):
    """fixed randomness"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_worker_seed(worker_id):
    """fixed cpu random pick images"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def find_free_port():
    """หา port อัตโนมัติไว้ให้ gpu ใน node ใช้คุยกัน """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return str(s.getsockname()[1])


def is_dist():
    """check ว่าใช้ torchrun และมีหลาย gpu มั้ย"""
    return dist.is_available() and dist.is_initialized()


def get_rank():
    """ถามว่าเป็น gpu ตัวไหน 0->n"""
    return dist.get_rank() if is_dist() else 0


def dist_barrier():
    """จุดนัดพบรอ gpu ตัวอื่น บอกว่าทำงานเร็จแล้วเดี๋ยวมารอทำงานต่อไปพร้อมกันที่จุดนี้"""
    if is_dist():
        dist.barrier()


def dist_all_ok(flag): # gpu ทำงานเสร็จส่ง flag มาบอก
    """check ว่าทำงานสำเร็จกันทุก gpu มั้ย"""
    if not is_dist():
        return bool(flag)
    t = torch.tensor([1 if flag else 0], dtype=torch.int32) # แปลงเป็น tensor([0]) / tensor([1])
    dist.all_reduce(t, op=dist.ReduceOp.MIN)    # เอาค่าในกล่อง t ของแต่ละ gpu มาทำ op แล้ว assign ค่ากลับเข้ากล่อง t เช่น [1, 1, 1, 0] MIN = 0 ทุก gpu t.item() = 0
                                                # ทำให้ gpu อื่นรู้สถานะของตัวเอง
    return bool(t.item()) 


def dist_broadcast_bool(flag, src=0): # src = rank
    """ให้ใช้ value จาก master rank"""
    if not is_dist():
        return bool(flag)
    t = torch.tensor([1 if flag else 0], dtype=torch.int32)
    dist.broadcast(t, src=src)  # assign ค่าของ master ไปให้ทุก gpu
    return bool(t.item())


def safe_torch_save(obj, path, retries=3, retry_delay=2.0):
    """มีไว้ป้องกันปัญหาที่อาจเกิดขึ้นได้เวลา save ข้อมูล ใช้เทคนิคที่เรียกว่า atomic write คือ ไม่ save ผ่าน network save ลง node storage แล้ว copy ผ่านทาง local storage(NAS)"""
    # save destination path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    local_tmp_dir = Path(os.environ.get('SLURM_TMPDIR') or os.environ.get('TMPDIR') or '/tmp')  # temp folder on node server
    local_tmp_dir.mkdir(parents=True, exist_ok=True)
 
    last_err = None
    for attempt in range(1, retries + 1):
        local_tmp_path = None   # node storage
        dest_tmp_path = None    # NAS storage
        try:
            fd, local_tmp_name = tempfile.mkstemp(dir=local_tmp_dir, suffix='.pt.tmp')  # create temp file
            os.close(fd)
            local_tmp_path = Path(local_tmp_name)
 
            torch.save(obj, local_tmp_path) # save on node storage
 
            dest_tmp_path = path.with_name(f'{path.name}.tmp{os.getpid()}')
            shutil.copyfile(local_tmp_path, dest_tmp_path) # copy from node storage to NAS
            os.replace(dest_tmp_path, path) # atomic within the destination filesystem(swap suffix)
            return True
        except Exception as e:
            last_err = e
            print(f"⚠️ safe_torch_save attempt {attempt}/{retries} for {path.name} failed: {e}")
            if dest_tmp_path is not None and dest_tmp_path.exists():
                dest_tmp_path.unlink(missing_ok=True)
            time.sleep(retry_delay)
        finally:
            if local_tmp_path is not None and local_tmp_path.exists():  # remove node temp dir
                local_tmp_path.unlink(missing_ok=True)
 
    print(f"❌ safe_torch_save giving up on {path} after {retries} attempts: {last_err}")
    return False
 
 
def atomic_json_dump(obj, path, indent=2):
    """ใช้หลักการ atomic save json file"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f'{path.name}.tmp{os.getpid()}')
    with open(tmp_path, 'w') as f:
        json.dump(obj, f, indent=indent, default=str)
    os.replace(tmp_path, path)
 
 
def append_csv_row(path, row, fieldnames):
    """write new data row into csv file"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0  # not exists or file size == 0
 
    # old file
    if not is_new:
        with open(path, 'r', newline='') as f:
            header = next(csv.reader(f), [])
        if header != list(fieldnames):  # columns field miss match
            raise ValueError(
                f"{path} was written with different columns.\n"
                f"  file   : {header}\n  expected: {list(fieldnames)}\n"
                f"Use a new experiment name (-e) or delete the old CSV."
            )
 
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def save_dcp_checkpoint(state_dict, checkpoint_dir, retries=3, retry_delay=2.0):
    """แนะนำให้ใช้แทน torch.save เวลาต้อง save model มี่มาจากการทำ DDP / FSDP 
    แก้ปัญหาการ save แบบเก่าคือเวลา save ปกติจะรวมของทุกอย่างมาที่ rank 0 ทำให้เกิด 
    OOM ด้วยการ save ของใครของมันลงใน folder เดียวกัน ทำให้ auto-sharding ได้ 
    เช่น เอาไปเทรนต่อด้วย 2 / 3 / 4 gpu"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)   # save folder
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            dcp.save(state_dict, checkpoint_id=str(checkpoint_dir)) # individual save
            return True
        except Exception as e:
            last_err = e
            print(f"⚠️ save_dcp_checkpoint attempt {attempt}/{retries} for {checkpoint_dir.name} failed: {e}")
            time.sleep(retry_delay)
    print(f"❌ save_dcp_checkpoint giving up on {checkpoint_dir} after {retries} attempts: {last_err}")
    return False


def load_dcp_checkpoint(state_dict, checkpoint_dir):
    """function for loading dcp state dict"""
    dcp.load(state_dict, checkpoint_id=str(checkpoint_dir))
    return state_dict


def dcp_model_to_pt(model, checkpoint_dir, out_path, is_main=True, retries=3, retry_delay=2.0):
    """convert dcp to pt ใช้รวมเป็น pt ไฟล์เดียว"""
    checkpoint_dir = Path(checkpoint_dir)
    out_path = Path(out_path)

    model_sd = get_model_state_dict(model)
    load_dcp_checkpoint({'model': model_sd}, checkpoint_dir)
    set_model_state_dict(model, model_sd)

    if not is_main:
        return True

    return safe_torch_save(model.state_dict(), out_path, retries=retries, retry_delay=retry_delay)