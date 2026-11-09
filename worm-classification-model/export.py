import torch
import torch.nn as nn
import torch.nn.functional as F
from models import build_model
from config import IMG_SIZE

class ModelPipeline(nn.Module):
    def __init__(self, base_model, img_size=IMG_SIZE):
        super().__init__()
        self.model = base_model
        self.img_size = img_size
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def forward(self, x):
        # --- resize and pad ---
        if x.dim() == 3:
            x = x.unsqueeze(0)
            
        N, C, h, w = x.shape
        max_size = torch.max(torch.stack([h, w]))
        
        scale = self.img_size / max_size.float()
        do_resize = max_size > self.img_size
        
        new_h = torch.where(do_resize, torch.clamp(torch.round(h * scale), min=1), h).long()
        new_h = torch.min(torch.tensor(self.img_size, device=x.device), new_h)
        
        new_w = torch.where(do_resize, torch.clamp(torch.round(w * scale), min=1), w).long()
        new_w = torch.min(torch.tensor(self.img_size, device=x.device), new_w)
    
        if do_resize:
            x = F.interpolate(x, size=(int(new_h.item()) if not torch.jit.is_scripting() and isinstance(new_h, torch.Tensor) and new_h.numel()==1 and not new_h.requires_grad else new_h, 
                                      int(new_w.item()) if not torch.jit.is_scripting() and isinstance(new_w, torch.Tensor) and new_w.numel()==1 and not new_w.requires_grad else new_w), 
                              mode='bilinear', align_corners=False)
            
        pad_h = self.img_size - new_h
        pad_w = self.img_size - new_w
        
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        
        x = F.pad(x, (int(pad_left), int(pad_right), int(pad_top), int(pad_bottom)), value=0.0)
        
        # --- normalize ---
        x = x / 255.0   # uint8 to float
        x = (x- self.mean) / self.std   # normalize
        
        return self.model(x)
    
pt_weight = '/scratch/lt200264-saiwat/worm-project/worm-classification-model/background-benchmark/ckpt/2class_dim/convnext_large/convnext_large/best.pt'

base_model = build_model('convnext_large', 2, pt_weight)
base_model.eval()

model = ModelPipeline(base_model)
model.eval()

dummy_input = (torch.randn(1, 3, 224, 224), ) # incase there are more than 1 input
export_path = '/scratch/lt200264-saiwat/worm-project/worm-classification-model/onnx-models/convnext_l_2cls.onnx'

print('[Log] Start exporting...')

try:
    torch.onnx.export(
        model,
        dummy_input,
        f=export_path,
        verbose=True,
        opset_version=17,
        dynamo=True,
        dynamic_shapes={    # define shapes that are not fixed
            'input': {
                0: torch.export.Dim('batch'),
                2: torch.export.Dim('height'),
                3: torch.export.Dim('width'),
            }
        }
    )
    
    print(f'[Log] Export sucess. Onnx model locates at {export_path}')

except Exception as e:
    print(f'[Error] Export fail.\n{e}')