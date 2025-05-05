'''
B. Adriano, N. Yokoya, J. Xia, H. Miura, W. Liu, M. Matsuoka, and S. Koshimura,
“Learning from multimodal and multitemporal earth observation data for building damage mapping,”
ISPRS J. Photogramm. Remote Sens., vol. 175, pp. 132–143, 2021
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.conv = ConvBlock(out_channels, out_channels)
        
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.fc_2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x, r):
        avg_out = self.fc_1(self.global_avg_pool(r))
        max_out = self.fc_2(self.global_max_pool(r))
        return x + x * (avg_out + max_out)

class SiamAttnUNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SiamAttnUNet, self).__init__()
        # Pre-disaster encoder
        self.encoder1_pre = ConvBlock(in_channels, 64)
        self.encoder2_pre = ConvBlock(64, 128)
        self.encoder3_pre = ConvBlock(128, 256)
        self.encoder4_pre = ConvBlock(256, 512)
        self.encoder5_pre = ConvBlock(512, 1024)
        
        # Post-disaster encoder
        self.encoder1_post = ConvBlock(in_channels, 64)
        self.encoder2_post = ConvBlock(64, 128)
        self.encoder3_post = ConvBlock(128, 256)
        self.encoder4_post = ConvBlock(256, 512)
        self.encoder5_post = ConvBlock(512, 1024)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fuse_1 = nn.Conv2d(in_channels=64 * 2, out_channels = 64, kernel_size=1)
        self.fuse_2 = nn.Conv2d(in_channels=128 * 2, out_channels = 128, kernel_size=1)
        self.fuse_3 = nn.Conv2d(in_channels=256 * 2, out_channels = 256, kernel_size=1)
        self.fuse_4 = nn.Conv2d(in_channels=512 * 2, out_channels = 512, kernel_size=1)
        self.fuse_5 = nn.Conv2d(in_channels=1024 * 2, out_channels = 1024, kernel_size=1)

        self.attention1 = ChannelAttention(512)
        self.attention2 = ChannelAttention(256)
        self.attention3 = ChannelAttention(128)
        self.attention4 = ChannelAttention(64)
        
        self.upconv1 = UpConv(1024, 512)
        self.upconv2 = UpConv(512 * 2, 256)
        self.upconv3 = UpConv(256 * 2, 128)
        self.upconv4 = UpConv(128 * 2, 64)
        self.upconv5 = ConvBlock(64 * 2, 64) # UpConv(, 64)

        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x1, x2):
        # Pre-disaster feature extraction
        enc1_1 = self.encoder1_pre(x1)
        enc2_1 = self.encoder2_pre(self.pool(enc1_1))
        enc3_1 = self.encoder3_pre(self.pool(enc2_1))
        enc4_1 = self.encoder4_pre(self.pool(enc3_1))
        enc5_1 = self.encoder5_pre(self.pool(enc4_1))
        
        # Post-disaster feature extraction
        enc1_2 = self.encoder1_post(x2)
        enc2_2 = self.encoder2_post(self.pool(enc1_2))
        enc3_2 = self.encoder3_post(self.pool(enc2_2))
        enc4_2 = self.encoder4_post(self.pool(enc3_2))
        enc5_2 = self.encoder5_post(self.pool(enc4_2))
        
        enc_1 = self.fuse_1(torch.cat([enc1_1, enc1_2], dim=1))
        enc_2 = self.fuse_2(torch.cat([enc2_1, enc2_2], dim=1))
        enc_3 = self.fuse_3(torch.cat([enc3_1, enc3_2], dim=1))
        enc_4 = self.fuse_4(torch.cat([enc4_1, enc4_2], dim=1))
        enc_5 = self.fuse_5(torch.cat([enc5_1, enc5_2], dim=1))

        up1 = self.upconv1(enc_5)

        enc_4 = self.attention1(x=enc_4, r=up1)
        up2 = self.upconv2(torch.cat([up1, enc_4], dim=1))

        enc_3 = self.attention2(x=enc_3, r=up2)
        up3 = self.upconv3(torch.cat([up2, enc_3], dim=1))

        enc_2 = self.attention3(x=enc_2, r=up3)
        up4 = self.upconv4(torch.cat([up3, enc_2], dim=1))

        enc_1 = self.attention4(x=enc_1, r=up4)
        up5 = self.upconv5(torch.cat([up4, enc_1], dim=1))
        
        output = self.final(up5)
        return output
    