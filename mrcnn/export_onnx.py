import torch
from config import ONNX_OPSET_VER


def export_onnx(pt_path, onnx_path, min_size, max_size, build_model_fn, device, opset_version=ONNX_OPSET_VER):
    """build_model_fn(num_cls, pt_path, device) -> nn.Module, already loaded with the
    checkpoint's weights. Keeping model construction as an injected callback means
    this module doesn't need to know about anchors/RPN/box hyperparameters."""
    print("--- Exporting PyTorch to ONNX ---")
    print(f"Loading checkpoint: {pt_path}")

    checkpoint = torch.load(pt_path, map_location=device, weights_only=True)
    cls_map = checkpoint["class_map"]
    num_cls = len(cls_map) + 1

    model = build_model_fn(num_cls, pt_path, device)
    model.eval()

    dummy_input = torch.randn(1, 3, min_size, max_size).to(device)

    print("Exporting...")
    torch.onnx.export(
        model,
        (dummy_input,),  # model อาจรับ input หลายตัวต้องส่งเป็น tuple
        onnx_path,
        opset_version=opset_version,
        do_constant_folding=True,  # optimization
        input_names=["images"],
        output_names=["boxes", "labels", "scores", "masks"],
        dynamic_axes={
            # dynamic input shape
            "images": {0: "batch_size", 2: "height", 3: "width"},
            # dynamic output
            "boxes": {0: "num_dets"},
            "labels": {0: "num_dets"},
            "scores": {0: "num_dets"},
            "masks": {0: "num_dets", 2: "mask_h", 3: "mask_w"},
        },
    )
    print(f"Successfully exported to: {onnx_path}\n")