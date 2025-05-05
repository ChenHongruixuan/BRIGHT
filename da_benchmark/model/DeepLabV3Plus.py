import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPP, self).__init__()
        self.atrous_convs = nn.ModuleList()
        for rate in atrous_rates:
            self.atrous_convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False)
            )
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.final_conv = nn.Conv2d(out_channels * (len(atrous_rates) + 2), out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        size = x.shape[2:]
        res = [self.relu(self.batch_norm(self.conv1x1(x)))]
        for conv in self.atrous_convs:
            res.append(self.relu(self.batch_norm(conv(x))))
        res.append(F.interpolate(self.global_avg_pool(x), size=size, mode='bilinear', align_corners=False))
        res = torch.cat(res, dim=1)
        return self.final_conv(res)


class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels=3, num_classes=4, atrous_rates=[6, 12, 18], output_stride=16):
        super(DeepLabV3Plus, self).__init__()
        
        # Load a pre-trained ResNet-50 model
        self.backbone = models.resnet50(pretrained=True)
        
        # Modify the first convolutional layer to accept different number of input channels
        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        low_level_channels = 256
        high_level_channels = 2048

        if output_stride == 16:
            self.backbone.layer4[0].conv2.stride = (1, 1)
            self.backbone.layer4[0].downsample[0].stride = (1, 1)
            self.aspp = ASPP(high_level_channels, 256, atrous_rates)
        else:
            raise NotImplementedError

        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[2:]
        x1 = self.backbone.conv1(x)
        x1 = self.backbone.bn1(x1)
        x1 = self.backbone.relu(x1)
        x1 = self.backbone.maxpool(x1)

        x2 = self.backbone.layer1(x1)
        x3 = self.backbone.layer2(x2)
        x4 = self.backbone.layer3(x3)
        x5 = self.backbone.layer4(x4)

        low_level_features = self.low_level_conv(x2)
        high_level_features = self.aspp(x5)
        high_level_features = F.interpolate(high_level_features, size=low_level_features.shape[2:], mode='bilinear', align_corners=False)

        concatenated_features = torch.cat((low_level_features, high_level_features), dim=1)
        x = self.decoder(concatenated_features)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)

        return x
