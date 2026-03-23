"""Mask R-CNN model with 4-channel fusion input (pre-event RGB + post-event SAR)."""

import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def build_model(
    num_classes: int = 4,
    pretrained: bool = True,
    pixel_mean: list = None,
    pixel_std: list = None,
    box_detections_per_img: int = 1500,
    rpn_pre_nms_top_n_test: int = 1500,
    rpn_post_nms_top_n_test: int = 1500,
) -> nn.Module:
    """Build a Mask R-CNN model with 4-channel input for RGB+SAR fusion.

    The first three channels correspond to pre-event RGB and the fourth to
    post-event SAR.  Conv1 is replaced with a 4-channel variant: the first
    three channels copy pretrained ImageNet weights and the fourth is
    initialised with the mean of the RGB kernel weights.

    Args:
        num_classes: Number of classes including background (default 4).
        pretrained: If True, load ImageNet-pretrained weights.
        pixel_mean: Per-channel mean [R, G, B, SAR] for normalisation.
        pixel_std: Per-channel std [R, G, B, SAR] for normalisation.
        box_detections_per_img: Maximum detections per image.
        rpn_pre_nms_top_n_test: Proposals kept before RPN NMS during testing.
        rpn_post_nms_top_n_test: Proposals kept after RPN NMS during testing.
    """
    image_mean = list(pixel_mean) if pixel_mean is not None else None
    image_std = list(pixel_std) if pixel_std is not None else None

    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn(
        weights=weights,
        image_mean=image_mean,
        image_std=image_std,
        box_detections_per_img=box_detections_per_img,
        rpn_pre_nms_top_n_test=rpn_pre_nms_top_n_test,
        rpn_post_nms_top_n_test=rpn_post_nms_top_n_test,
    )

    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace mask predictor
    old_mask_pred = model.roi_heads.mask_predictor
    in_features_mask = old_mask_pred.conv5_mask.in_channels
    dim_reduced = old_mask_pred.conv5_mask.out_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, dim_reduced, num_classes
    )

    # Replace conv1: 3-channel -> 4-channel (RGB + SAR)
    old_conv1 = model.backbone.body.conv1
    old_weight = old_conv1.weight.data  # [out_ch, 3, kH, kW]

    new_conv1 = nn.Conv2d(
        in_channels=4,
        out_channels=old_conv1.out_channels,
        kernel_size=old_conv1.kernel_size,
        stride=old_conv1.stride,
        padding=old_conv1.padding,
        dilation=old_conv1.dilation,
        groups=old_conv1.groups,
        bias=old_conv1.bias is not None,
        padding_mode=old_conv1.padding_mode,
    )
    # First 3 channels: pretrained RGB weights; 4th channel: mean of RGB weights
    new_conv1.weight.data[:, :3, :, :] = old_weight
    new_conv1.weight.data[:, 3:, :, :] = old_weight.mean(dim=1, keepdim=True)
    if old_conv1.bias is not None:
        new_conv1.bias.data = old_conv1.bias.data.clone()

    model.backbone.body.conv1 = new_conv1

    return model
