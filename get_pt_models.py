import os
import torch
import torchvision.models as models

# Local base directory for staging the files before SCP
BASE_DIR = "./pt-models"

# Map the torchvision model names to your desired local folder structure
TORCHVISION_MODELS = {
    # ConvNeXt
    "convnext_base": f"{BASE_DIR}/convnext/convnext_base.pth",
    "convnext_large": f"{BASE_DIR}/convnext/convnext_large.pth",
    "convnext_small": f"{BASE_DIR}/convnext/convnext_small.pth",
    "convnext_tiny": f"{BASE_DIR}/convnext/convnext_tiny.pth",

    # EfficientNetV2
    "efficientnet_v2_l": f"{BASE_DIR}/efficientnet_v2/efficientnet_v2_l.pth",
    "efficientnet_v2_m": f"{BASE_DIR}/efficientnet_v2/efficientnet_v2_m.pth",
    "efficientnet_v2_s": f"{BASE_DIR}/efficientnet_v2/efficientnet_v2_s.pth",

    # ResNet
    "resnet101": f"{BASE_DIR}/resnet/resnet101.pth",
    "resnet152": f"{BASE_DIR}/resnet/resnet152.pth",
    "resnet18": f"{BASE_DIR}/resnet/resnet18.pth",
    "resnet34": f"{BASE_DIR}/resnet/resnet34.pth",
    "resnet50": f"{BASE_DIR}/resnet/resnet50.pth",

    # Swin Transformer
    "swin_b": f"{BASE_DIR}/swin/swin_b.pth",
    "swin_s": f"{BASE_DIR}/swin/swin_s.pth",
    "swin_t": f"{BASE_DIR}/swin/swin_t.pth",
    "swin_v2_b": f"{BASE_DIR}/swin/swin_v2_b.pth",
    "swin_v2_s": f"{BASE_DIR}/swin/swin_v2_s.pth",
    "swin_v2_t": f"{BASE_DIR}/swin/swin_v2_t.pth",

    # Vision Transformer (ViT)
    "vit_b_16": f"{BASE_DIR}/vit/vit_b_16.pth",
    "vit_b_32": f"{BASE_DIR}/vit/vit_b_32.pth",
    "vit_h_14": f"{BASE_DIR}/vit/vit_h_14.pth",
    "vit_l_16": f"{BASE_DIR}/vit/vit_l_16.pth",
    "vit_l_32": f"{BASE_DIR}/vit/vit_l_32.pth",
}

def download_and_save_weights():
    print(f"Creating local staging directory at: {BASE_DIR}\n")
    
    for model_name, save_path in TORCHVISION_MODELS.items():
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if os.path.exists(save_path):
            print(f"[SKIPPED] {model_name} -> Already exists at {save_path}")
            continue
            
        print(f"[DOWNLOADING] {model_name}...")
        
        try:
            model_func = getattr(models, model_name)
            model = model_func(weights='DEFAULT')
            torch.save(model.state_dict(), save_path)
            
            print(f"  [SUCCESS] Saved to {save_path}")
            
        except AttributeError:
            print(f"  [ERROR] Model '{model_name}' not found. Check torchvision version.")
        except Exception as e:
            print(f"  [ERROR] Failed to download {model_name}: {e}")

    print("\nDownload complete! You can now copy the folder to your remote server using:")

if __name__ == "__main__":
    download_and_save_weights()