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
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_worker_seed(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return str(s.getsockname()[1])


def safe_torch_save(obj, path, retries=3, retry_delay=2.0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    local_tmp_dir = Path(os.environ.get('SLURM_TMPDIR') or os.environ.get('TMPDIR') or '/tmp')
    local_tmp_dir.mkdir(parents=True, exist_ok=True)

    last_err = None
    for attempt in range(1, retries + 1):
        local_tmp_path = None
        dest_tmp_path = None
        try:
            fd, local_tmp_name = tempfile.mkstemp(dir=local_tmp_dir, suffix='.pt.tmp')
            os.close(fd)
            local_tmp_path = Path(local_tmp_name)

            torch.save(obj, local_tmp_path)

            dest_tmp_path = path.with_name(f'{path.name}.tmp{os.getpid()}')
            shutil.copyfile(local_tmp_path, dest_tmp_path)
            os.replace(dest_tmp_path, path)  # atomic within the destination filesystem
            return True
        except Exception as e:
            last_err = e
            print(f"⚠️ safe_torch_save attempt {attempt}/{retries} for {path.name} failed: {e}")
            if dest_tmp_path is not None and dest_tmp_path.exists():
                dest_tmp_path.unlink(missing_ok=True)
            time.sleep(retry_delay)
        finally:
            if local_tmp_path is not None and local_tmp_path.exists():
                local_tmp_path.unlink(missing_ok=True)

    print(f"❌ safe_torch_save giving up on {path} after {retries} attempts: {last_err}")
    return False


def atomic_json_dump(obj, path, indent=2):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f'{path.name}.tmp{os.getpid()}')
    with open(tmp_path, 'w') as f:
        json.dump(obj, f, indent=indent, default=str)
    os.replace(tmp_path, path)


def append_csv_row(path, row, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def save_dcp_checkpoint(state_dict, checkpoint_dir, retries=3, retry_delay=2.0):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            dcp.save(state_dict, checkpoint_id=str(checkpoint_dir))
            return True
        except Exception as e:
            last_err = e
            print(f"⚠️ save_dcp_checkpoint attempt {attempt}/{retries} for {checkpoint_dir.name} failed: {e}")
            time.sleep(retry_delay)
    print(f"❌ save_dcp_checkpoint giving up on {checkpoint_dir} after {retries} attempts: {last_err}")
    return False


def load_dcp_checkpoint(state_dict, checkpoint_dir):
    dcp.load(state_dict, checkpoint_id=str(checkpoint_dir))
    return state_dict


def dcp_model_to_pt(model, checkpoint_dir, out_path, is_main=True, retries=3, retry_delay=2.0):
    checkpoint_dir = Path(checkpoint_dir)
    out_path = Path(out_path)

    model_sd = get_model_state_dict(model)
    load_dcp_checkpoint({'model': model_sd}, checkpoint_dir)
    set_model_state_dict(model, model_sd)

    if not is_main:
        return True

    return safe_torch_save(model.state_dict(), out_path, retries=retries, retry_delay=retry_delay)