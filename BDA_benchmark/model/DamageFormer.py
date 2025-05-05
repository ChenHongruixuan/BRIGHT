'''
Hongruixuan Chen, Edoardo Nemni, Sofia Vallecorsa, Xi Li, Chen Wu, and Lars Bromley, 
"Dual-Tasks Siamese Transformer Framework for Building Damage Assessment," 
2022 IEEE International Geoscience and Remote Sensing Symposium (IGARSS), Kuala Lumpur, Malaysia, 2022, pp. 1600-1603
'''

import torch
import torch.nn.functional as F

import torch
import torch.nn as nn

from typing import Optional, Callable, Any


from torchvision.models import swin_t, Swin_T_Weights
import torch.nn as nn
import torch
import torch.nn.functional as F


class SwinTransformerFeatureExtractor(nn.Module):
    def __init__(self, swin_transformer, depths):
        super().__init__()
        self.swin_transformer = swin_transformer
        self.depths = depths
        # print(self.swin_transformer.features)

    def forward(self, x):
        feature_maps = []

        for layer in self.swin_transformer.features:
            x = layer(x)
            # Directly capture the output after each Sequential block of SwinTransformerBlock instances
            if isinstance(layer, nn.Sequential):
                feature_maps.append(x)

        # Now, feature_maps contains the outputs right after each Sequential block,
        # which are essentially the end-of-stage features you're interested in.
        return feature_maps

   
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out 


class DamageFormer(nn.Module):
    def __init__(self):
        super(DamageFormer, self).__init__()

        # if it is multimodal task (if not, please use pure-siamese architecture)
        self.encoder_1 = SwinTransformerFeatureExtractor(swin_transformer = swin_t(weights=Swin_T_Weights), depths=[2, 2, 6, 2]) # getattr(mix_transformer, backbone)()
        self.encoder_2 = SwinTransformerFeatureExtractor(swin_transformer = swin_t(weights=Swin_T_Weights), depths=[2, 2, 6, 2]) # getattr(mix_transformer, backbone)()

        self.fusion_layer_1 = ResBlock(in_channels=1440, out_channels=256, stride=1, downsample=nn.Conv2d(in_channels=1440, out_channels=256, kernel_size=1))
        self.fusion_layer_2 = ResBlock(in_channels=1440, out_channels=256, stride=1, downsample=nn.Conv2d(in_channels=1440, out_channels=256, kernel_size=1))
        self.fusion_layer_3 = ResBlock(in_channels=512, out_channels=256, stride=1, downsample=nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1))
       
        
        self.clf_1 = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1)
        self.clf_2 = nn.Conv2d(in_channels=256, out_channels=4, kernel_size=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y
    
    def forward(self, pre_data, post_data):
        _, pre_low_level_feat_1, pre_low_level_feat_2, pre_low_level_feat_3, pre_output = \
            self.encoder_1(pre_data)
        _, post_low_level_feat_1, post_low_level_feat_2, post_low_level_feat_3, post_output = \
            self.encoder_2(post_data)
        
        pre_low_level_feat_1 = pre_low_level_feat_1.permute(0, 3, 1, 2)
        post_low_level_feat_1 = post_low_level_feat_1.permute(0, 3, 1, 2)
        pre_low_level_feat_2 = pre_low_level_feat_2.permute(0, 3, 1, 2)
        post_low_level_feat_2 = post_low_level_feat_2.permute(0, 3, 1, 2)
        pre_low_level_feat_3 = pre_low_level_feat_3.permute(0, 3, 1, 2)
        post_low_level_feat_3 = post_low_level_feat_3.permute(0, 3, 1, 2)
        pre_output = pre_output.permute(0, 3, 1, 2)
        post_output = post_output.permute(0, 3, 1, 2)

        p41 = F.interpolate(pre_output, size=pre_low_level_feat_1.size()[2:], mode='bilinear')
        p42 = F.interpolate(post_output, size=post_low_level_feat_1.size()[2:], mode='bilinear')

        p31 = F.interpolate(pre_low_level_feat_3, size=pre_low_level_feat_1.size()[2:], mode='bilinear')
        p32 = F.interpolate(post_low_level_feat_3, size=post_low_level_feat_1.size()[2:], mode='bilinear')
       

        p21 = F.interpolate(pre_low_level_feat_2, size=pre_low_level_feat_1.size()[2:], mode='bilinear')
        p22 = F.interpolate(post_low_level_feat_2, size=post_low_level_feat_1.size()[2:], mode='bilinear')
        

        feat_1 = torch.cat([p41, p31, p21, pre_low_level_feat_1], dim=1)
        feat_2 = torch.cat([p42, p32, p22, post_low_level_feat_1], dim=1)

        final_feat_1 = self.fusion_layer_1(feat_1)
        feat_2 = self.fusion_layer_2(feat_2)

        final_feat_2 = torch.cat([final_feat_1, feat_2], dim=1)
        final_feat_2 = self.fusion_layer_3(final_feat_2)

        output_loc =  self.clf_1(final_feat_1)
        output_loc = F.interpolate(output_loc, size=pre_data.size()[-2:], mode='bilinear')
        output_dam =  self.clf_2(final_feat_2)
        output_dam = F.interpolate(output_dam, size=post_data.size()[-2:], mode='bilinear')
        return output_loc, output_dam 