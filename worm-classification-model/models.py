import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from ultralytics import YOLO
import config as c

MODEL_REGISTRY = {
    'resnet18': (models.resnet18, c.RESNET18_PATH),
    'resnet34': (models.resnet34, c.RESNET34_PATH),
    'resnet50': (models.resnet50, c.RESNET50_PATH),
    'resnet101': (models.resnet101, c.RESNET101_PATH),
    'resnet152': (models.resnet152, c.RESNET152_PATH),
    'convnext_tiny': (models.convnext_tiny, c.CONVNEXT_TINY_PATH),
    'convnext_small': (models.convnext_small, c.CONVNEXT_SMALL_PATH),
    'convnext_base': (models.convnext_base, c.CONVNEXT_BASE_PATH),
    'convnext_large': (models.convnext_large, c.CONVNEXT_LARGE_PATH),
    'efficientnet_v2_s': (models.efficientnet_v2_s, c.EFFICIENTNET_V2_S_PATH),
    'efficientnet_v2_m': (models.efficientnet_v2_m, c.EFFICIENTNET_V2_M_PATH),
    'efficientnet_v2_l': (models.efficientnet_v2_l, c.EFFICIENTNET_V2_L_PATH),
    'vit_b_16': (models.vit_b_16, c.VIT_B_16_PATH),
    'vit_b_32': (models.vit_b_32, c.VIT_B_32_PATH),
    'vit_l_16': (models.vit_l_16, c.VIT_L_16_PATH),
    'vit_l_32': (models.vit_l_32, c.VIT_L_32_PATH),
    'vit_h_14': (models.vit_h_14, c.VIT_H_14_PATH),
    'swin_t': (models.swin_t, c.SWIN_T_PATH),
    'swin_s': (models.swin_s, c.SWIN_S_PATH),
    'swin_b': (models.swin_b, c.SWIN_B_PATH),
    'swin_v2_t': (models.swin_v2_t, c.SWIN_V2_T_PATH),
    'swin_v2_s': (models.swin_v2_s, c.SWIN_V2_S_PATH),
    'swin_v2_b': (models.swin_v2_b, c.SWIN_V2_B_PATH),
    'yolo26n': (None, c.YOLO26N_CLS_PATH),
    'yolo26s': (None, c.YOLO26S_CLS_PATH),
    'yolo26m': (None, c.YOLO26M_CLS_PATH),
    'yolo26l': (None, c.YOLO26L_CLS_PATH),
    'yolo26x': (None, c.YOLO26X_CLS_PATH),
}

FAMILY_PREFIXES = sorted(
    ['efficientnet_v2', 'convnext', 'swin_v2', 'swin', 'vit', 'resnet', 'yolo'], key=len, reverse=True
)


def get_model_family(model_name):
    for prefix in FAMILY_PREFIXES:
        if model_name.startswith(prefix):
            return prefix
    return model_name


def resolve_model_names(model_name):
    model_name = model_name.lower()
    if model_name in MODEL_REGISTRY:
        return [model_name]
    matches = [k for k in MODEL_REGISTRY if k.startswith(model_name)]
    if not matches:
        raise ValueError(
            f"Unknown model name or family '{model_name}'.\n"
            f"Available options: {list(MODEL_REGISTRY.keys())}"
        )
    return matches


def _load_yolo_cls_backbone(ckpt_path):
    wrapper = YOLO(ckpt_path)   # load architecture + weights
    model = wrapper.model       # ทำให้ใช้แบบ nn.Module ได้
    
    model.train()
    for p in model.parameters():
        p.requires_grad = True
        
    return model


def _change_model_head(model, model_name, num_classes):
    if model_name.startswith('resnet'):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name.startswith('convnext'):
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
    elif model_name.startswith('efficientnet'):
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name.startswith('vit'):
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    elif model_name.startswith('swin'):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    elif model_name.startswith('yolo'):
        head = model.model[-1]  # classifier head
        in_features = head.linear.in_features
        head.linear = nn.Linear(in_features, num_classes)
    return model


def _interpolate_vit_pos_embedding(state_dict, model):
    key = 'encoder.pos_embedding'
    if key not in state_dict:
        return state_dict

    old_pos_embed = state_dict[key]
    new_pos_embed = model.encoder.pos_embedding

    if old_pos_embed.shape == new_pos_embed.shape:
        return state_dict

    num_extra_tokens = 1  # class token
    old_num_patches = old_pos_embed.shape[1] - num_extra_tokens
    new_num_patches = new_pos_embed.shape[1] - num_extra_tokens

    old_size = int(round(old_num_patches ** 0.5))
    new_size = int(round(new_num_patches ** 0.5))

    if old_size * old_size != old_num_patches or new_size * new_size != new_num_patches:
        raise ValueError(
            f"Cannot interpolate ViT position embedding: non-square patch grid "
            f"(old_patches={old_num_patches}, new_patches={new_num_patches})"
        )

    print(f"ℹ️  Interpolating ViT pos_embedding: {old_size}x{old_size} -> {new_size}x{new_size} grid")

    cls_pos = old_pos_embed[:, :num_extra_tokens]
    patch_pos = old_pos_embed[:, num_extra_tokens:]

    dim = patch_pos.shape[-1]
    patch_pos = patch_pos.reshape(1, old_size, old_size, dim).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(patch_pos, size=(new_size, new_size), mode='bicubic', align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, new_size * new_size, dim)

    new_full_pos = torch.cat([cls_pos, patch_pos], dim=1)

    state_dict = dict(state_dict)
    state_dict[key] = new_full_pos
    return state_dict


class _TupleOutputUnwrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out[0] if isinstance(out, tuple) else out
    
    
def build_model(model_name, num_classes, finetune_path=None, device=c.DEVICE):
    model_name = model_name.lower()
    model_function, checkpoint_path = MODEL_REGISTRY[model_name]

    if model_name.startswith('yolo'):
        model = _load_yolo_cls_backbone(checkpoint_path)
    else:
        model = model_function(weights=None)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        state_dict = checkpoint.get('state_dict', checkpoint)

        if model_name.startswith('vit'):
            state_dict = _interpolate_vit_pos_embedding(state_dict, model)

        model.load_state_dict(state_dict, strict=True)

    model = _change_model_head(model, model_name, num_classes)

    if model_name.startswith('yolo'):
        model = _TupleOutputUnwrapper(model)

    if finetune_path is not None:
        checkpoint = torch.load(finetune_path, map_location=device, weights_only=True)
        state_dict = checkpoint.get('state_dict', checkpoint)
        if model_name.startswith('vit'):
            state_dict = _interpolate_vit_pos_embedding(state_dict, model)
        model.load_state_dict(state_dict, strict=True)

    model.to(device)
    return model
