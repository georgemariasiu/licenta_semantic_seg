import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOv2LinearSegmentation(nn.Module):

    def __init__(self, num_classes: int):

        super().__init__()

        self.encoder = torch.hub.load(
            repo_or_dir='facebookresearch/dinov2',
            model='dinov2_vitb14',
            pretrained=True,
        )

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.linear_head = nn.Conv2d(
            in_channels=768,
            out_channels=num_classes,
            kernel_size=1,
        )

    def train(self, mode: bool = True):

        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, H, W = x.shape

        with torch.no_grad():
            features = self.encoder.forward_features(x)

        patch_tokens = features['x_norm_patchtokens']  # (B, num_patches, 768)

        patch_size = self.encoder.patch_size
        h_patches = H // patch_size
        w_patches = W // patch_size

        # (B, num_patches, 768) -> (B, 768, num_patches) -> (B, 768, H, W)
        patch_tokens = patch_tokens.permute(0, 2, 1)
        patch_tokens = patch_tokens.reshape(B, 768, h_patches, w_patches)

        logits = self.linear_head(patch_tokens)

        logits = F.interpolate(
            logits,
            size=(H, W),
            mode='bilinear',
            align_corners=False,
        )

        return logits  # (B, num_classes, H, W)

def create_model(num_classes: int) -> DINOv2LinearSegmentation:

    return DINOv2LinearSegmentation(num_classes=num_classes)