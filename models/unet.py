# # models/unet.py

# import torch
# import torch.nn as nn


# class DoubleConv(nn.Module):
#     """2回の畳み込みを行うユニット"""

#     def __init__(self, in_channels, out_channels):
#         super(DoubleConv, self).__init__()
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),  # バッチ正規化（オプション）
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),  # バッチ正規化（オプション）
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         return self.double_conv(x)


# class UNet(nn.Module):
#     """U-Netモデル"""

#     def __init__(self, n_channels=3, n_classes=1):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes

#         # エンコーダ（ダウンサンプリング部）
#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
#         self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
#         self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
#         self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 512))

#         # デコーダ（アップサンプリング部）
#         self.up1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
#         self.conv_up1 = DoubleConv(512 + 512, 512)

#         self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
#         self.conv_up2 = DoubleConv(256 + 256, 256)

#         self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
#         self.conv_up3 = DoubleConv(128 + 128, 128)

#         self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#         self.conv_up4 = DoubleConv(64 + 64, 64)

#         # 出力層
#         self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

#     def forward(self, x):
#         x1 = self.inc(x)  # [B, 64, H, W]
#         x2 = self.down1(x1)  # [B, 128, H/2, W/2]
#         x3 = self.down2(x2)  # [B, 256, H/4, W/4]
#         x4 = self.down3(x3)  # [B, 512, H/8, W/8]
#         x5 = self.down4(x4)  # [B, 512, H/16, W/16]

#         # アップサンプリングとスキップ接続
#         x = self.up1(x5)  # [B, 512, H/8, W/8]
#         x = torch.cat([x, x4], dim=1)
#         x = self.conv_up1(x)

#         x = self.up2(x)  # [B, 256, H/4, W/4]
#         x = torch.cat([x, x3], dim=1)
#         x = self.conv_up2(x)

#         x = self.up3(x)  # [B, 128, H/2, W/2]
#         x = torch.cat([x, x2], dim=1)
#         x = self.conv_up3(x)

#         x = self.up4(x)  # [B, 64, H, W]
#         x = torch.cat([x, x1], dim=1)
#         x = self.conv_up4(x)

#         # 出力
#         logits = self.outc(x)  # [B, n_classes, H, W]
#         return logits


# models/unet.py

import torch
import torch.nn as nn


class TwoConvBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(TwoConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1)
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
        # ConvTranspose2dを使用してアップサンプリング
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.upconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UNet(nn.Module):
    """U-Netモデル"""

    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # エンコーダ（ダウンサンプリング部）
        self.TCB1 = TwoConvBlock(n_channels, 64, 64)
        self.TCB2 = TwoConvBlock(64, 128, 128)
        self.TCB3 = TwoConvBlock(128, 256, 256)
        self.TCB4 = TwoConvBlock(256, 512, 512)
        self.TCB5 = TwoConvBlock(512, 1024, 1024)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # デコーダ（アップサンプリング部）
        self.UC1 = UpConv(1024, 512)
        self.TCB6 = TwoConvBlock(1024, 512, 512)

        self.UC2 = UpConv(512, 256)
        self.TCB7 = TwoConvBlock(512, 256, 256)

        self.UC3 = UpConv(256, 128)
        self.TCB8 = TwoConvBlock(256, 128, 128)

        self.UC4 = UpConv(128, 64)
        self.TCB9 = TwoConvBlock(128, 64, 64)

        # 出力層
        self.conv1 = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # エンコーダ
        x1 = self.TCB1(x)
        x = self.maxpool(x1)

        x2 = self.TCB2(x)
        x = self.maxpool(x2)

        x3 = self.TCB3(x)
        x = self.maxpool(x3)

        x4 = self.TCB4(x)
        x = self.maxpool(x4)

        x5 = self.TCB5(x)

        # デコーダ
        x = self.UC1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.TCB6(x)

        x = self.UC2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.TCB7(x)

        x = self.UC3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.TCB8(x)

        x = self.UC4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.TCB9(x)

        # 出力層
        x = self.conv1(x)
        return x
