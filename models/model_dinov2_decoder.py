# UNET, ERFNET

import torch
import torch.nn as nn
import torch.nn.functional as F

# block with 6 layers
class DoubleConv(nn.Module):

    def __init__(self, in_ch: int, out_ch: int):

        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        return self.block(x)

# summary of in_ch features into out_ch
class Reassemble(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, scale: float):

        super().__init__()
        self.scale = scale
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):

        x = self.proj(x)
        if self.scale != 1.0:
            x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)

        return x

# grows features spatially by concatenating with feature map state from encoder
class UpBlock(nn.Module):

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):

        super().__init__()
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):

        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DINOv2UNetDecoder(nn.Module):

    def __init__(self, num_classes: int, tap_layers=(2, 5, 8, 11)):

        super().__init__()

        self.encoder = torch.hub.load(
            repo_or_dir='facebookresearch/dinov2',
            model='dinov2_vitb14',
            pretrained=True,
        )
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.tap_layers = tap_layers
        embed_dim = self.encoder.embed_dim # 768 for ViT-B/14

        self.re_fine = Reassemble(embed_dim, 64, scale=4.0)
        self.re_mid = Reassemble(embed_dim, 128, scale=2.0)
        self.re_low = Reassemble(embed_dim, 256, scale=1.0)
        self.re_bottle = Reassemble(embed_dim, 512, scale=0.5)

        self.up0 = UpBlock(in_ch=512, skip_ch=256, out_ch=256)
        self.up1 = UpBlock(in_ch=256, skip_ch=128, out_ch=128)
        self.up2 = UpBlock(in_ch=128, skip_ch=64, out_ch=64)

        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def train(self, mode: bool = True):

        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x):

        H, W = x.shape[-2:]

        with torch.no_grad():
            taps = self.encoder.get_intermediate_layers(
                x, n=self.tap_layers, reshape=True, norm=True
            )
        t_fine, t_mid, t_low, t_deep = taps

        s_fine = self.re_fine(t_fine)
        s_mid = self.re_mid(t_mid)
        s_low = self.re_low(t_low)
        bottle = self.re_bottle(t_deep)

        x = self.up0(bottle, s_low)
        x = self.up1(x, s_mid)
        x = self.up2(x, s_fine)

        x = self.out_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return x # (B, num_classes, H, W)


def create_model(num_classes: int) -> DINOv2UNetDecoder:

    return DINOv2UNetDecoder(num_classes=num_classes)