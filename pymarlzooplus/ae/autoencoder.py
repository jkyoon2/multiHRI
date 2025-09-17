import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    CNN encoder that maps (B, C=27, H, W) → (B, latent_dim)
    - Uses AdaptiveAvgPool2d(1) to be agnostic to (H, W)
    - Input is expected as float in [0, 1] (external normalization recommended)
    """

    def __init__(self, in_channels: int = 27, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # (B,256,1,1)
            nn.Flatten(),              # (B,256)
            nn.Linear(256, latent_dim),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.backbone(x)


class Decoder(nn.Module):
    """
    CNN decoder that maps (B, latent_dim) → (B, C=27, H, W)
    - Requires target spatial size (H, W) at construction (use your layout's size)
    """

    def __init__(self, out_channels: int = 27, latent_dim: int = 64, target_hw: tuple = (6, 5)):
        super().__init__()
        self.out_channels = out_channels
        self.target_hw = target_hw

        # Start from (B,256,1,1) then upsample progressively to (H,W)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
        )

        # Use a light upsampling stack with nearest interpolation to reach target size
        self.conv_stack = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, z: th.Tensor) -> th.Tensor:
        x = self.fc(z)                     # (B,256)
        x = x.view(x.shape[0], 256, 1, 1)  # (B,256,1,1)
        # interpolate to target size before last convs for stable shapes
        x = F.interpolate(x, size=self.target_hw, mode="nearest")
        x = self.conv_stack(x)             # (B,C,H,W)
        return x


class AutoEncoder(nn.Module):
    """Wrapper module combining Encoder and Decoder for training."""

    def __init__(self, in_channels: int = 27, target_hw: tuple = (6, 5), latent_dim: int = 64):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, latent_dim=latent_dim)
        self.decoder = Decoder(out_channels=in_channels, latent_dim=latent_dim, target_hw=target_hw)

    def forward(self, x: th.Tensor) -> tuple:
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return z, x_rec


