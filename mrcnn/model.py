import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint_sequential
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator


def get_mrcnn_model(
    num_cls,
    device,
    min_size,
    max_size,
    anchor_sizes,
    anchor_aspect_ratios,
    rpn_pre_nms_top_n_train,
    rpn_pre_nms_top_n_test,
    rpn_post_nms_top_n_train,
    rpn_post_nms_top_n_test,
    rpn_nms_thresh,
    rpn_fg_iou_thresh,
    rpn_bg_iou_thresh,
    rpn_batch_size_per_image,
    rpn_positive_fraction,
    rpn_score_thresh,
    box_score_thresh,
    box_nms_thresh,
    box_detections_per_img,
    box_fg_iou_thresh,
    box_bg_iou_thresh,
    box_batch_size_per_image,
    box_positive_fraction,
    coco_pretrained_path=None,
    pt_path=None,
):
    # backbone resnet50 ทำ cnn 5 ชั้น C2->C3->C4->C5->C6 มี fpn 5 ชั้น P2->P3->P4->P5->P6
    formatted_sizes = tuple((s,) for s in anchor_sizes)
    formatted_ratios = (tuple(anchor_aspect_ratios),) * len(anchor_sizes)

    anchor_generator = AnchorGenerator(
        sizes=formatted_sizes,
        aspect_ratios=formatted_ratios,
    )

    model = maskrcnn_resnet50_fpn_v2(
        weights=None,
        min_size=min_size,
        max_size=max_size,
        # RPN parameters
        rpn_pre_nms_top_n_train=rpn_pre_nms_top_n_train,
        rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
        rpn_post_nms_top_n_train=rpn_post_nms_top_n_train,
        rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
        rpn_nms_thresh=rpn_nms_thresh,
        rpn_fg_iou_thresh=rpn_fg_iou_thresh,
        rpn_bg_iou_thresh=rpn_bg_iou_thresh,
        rpn_batch_size_per_image=rpn_batch_size_per_image,
        rpn_positive_fraction=rpn_positive_fraction,
        rpn_score_thresh=rpn_score_thresh,
        # Box parameters
        box_score_thresh=box_score_thresh,
        box_nms_thresh=box_nms_thresh,
        box_detections_per_img=box_detections_per_img,
        box_fg_iou_thresh=box_fg_iou_thresh,
        box_bg_iou_thresh=box_bg_iou_thresh,
        box_batch_size_per_image=box_batch_size_per_image,
        box_positive_fraction=box_positive_fraction,
    )
    model.rpn.anchor_generator = anchor_generator

    if pt_path is None:
        # Fresh run: load COCO-pretrained backbone, then swap in fresh heads for num_cls
        state_dict = torch.load(coco_pretrained_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)

        in_features_bbox = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features_bbox, num_cls)

        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_cls)
    else:
        # Resume/finetune: build heads first, then load the full state dict
        in_features_bbox = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features_bbox, num_cls)

        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_cls)

        checkpoint = torch.load(pt_path, map_location=device, weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint)  # support both raw and wrapped checkpoints
        model.load_state_dict(state_dict, strict=True)

    return model.to(device)


def enable_backbone_checkpointing(model, segments=2):
    body = model.backbone.body  # IntermediateLayerGetter over resnet
    for name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(body, name)  # nn.Sequential of Bottleneck blocks
        n_blocks = len(layer)
        segs = min(segments, n_blocks)

        orig_forward = layer.forward

        def make_ckpt_forward(seq, segs, orig_forward):
            def ckpt_forward(x):
                if not torch.is_grad_enabled():
                    return orig_forward(x)
                return checkpoint_sequential(seq, segs, x, use_reentrant=False)
            return ckpt_forward

        layer.forward = make_ckpt_forward(layer, segs, orig_forward)
    return model


def unwrap_model(model):
    """Return the underlying nn.Module whether or not it's DDP-wrapped."""
    return model.module if isinstance(model, DDP) else model