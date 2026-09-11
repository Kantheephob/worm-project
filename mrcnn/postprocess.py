import torch


def binarize_mask_preds(preds, threshold=0.5):
    binarized_preds = []
    for p in preds:
        new_p = p.copy()
        if "masks" in new_p:
            new_p["masks"] = (new_p["masks"].squeeze(1) > threshold)
        binarized_preds.append(new_p)

    return binarized_preds


def to_cpu_dicts(dict_list):
    cpu_dicts = []
    for d in dict_list:
        cpu_dict = {}
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                cpu_dict[k] = v.detach().cpu()
            else:
                cpu_dict[k] = v
        cpu_dicts.append(cpu_dict)

    return cpu_dicts