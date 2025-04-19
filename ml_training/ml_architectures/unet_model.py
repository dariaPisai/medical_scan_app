# ml_training/model_architectures/unet_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # Input channels to DoubleConv is in_channels (from previous layer) + out_channels (from skip connection)
            # because skip connection comes after upsampling but before conv
            self.conv = DoubleConv(in_channels + out_channels, out_channels, in_channels) # Adjusted input channels
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2) # Halve channels during upconv
             # Input channels to DoubleConv is in_channels ( halved then doubled) + out_channels from skip
            self.conv = DoubleConv(in_channels + out_channels, out_channels) # Adjusted input channels

    def forward(self, x1, x2): # x1 is from upsample path, x2 is skip connection
        x1 = self.up(x1)
        # input is NCHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # Pad x1 to match x2 size for concatenation
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1) # Concatenate along channel dimension
        return self.conv(x)

class OutConv(nn.Module):
    """Final 1x1 convolution"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        """
        Args:
            n_channels (int): Number of input channels (e.g., 1 for grayscale, 4 for BraTS sequences)
            n_classes (int): Number of output classes (e.g., 1 for binary mask, or 3 for BraTS tumor core, whole tumor, enhancing tumor)
            bilinear (bool): Whether to use bilinear upsampling or ConvTranspose2d
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024 // factor, 512 // factor, bilinear) # Pass skip connection features directly
        self.up2 = Up(512 // factor, 256 // factor, bilinear)
        self.up3 = Up(256 // factor, 128 // factor, bilinear)
        self.up4 = Up(128 // factor, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4) # Pass skip connection x4
        x = self.up2(x, x3)  # Pass skip connection x3
        x = self.up3(x, x2)  # Pass skip connection x2
        x = self.up4(x, x1)  # Pass skip connection x1
        logits = self.outc(x)
        # Note: Often you don't apply sigmoid/softmax here,
        # the loss function (e.g., BCEWithLogitsLoss, CrossEntropyLoss) handles it.
        return logits

# --- Alternative using MONAI ---
# If you installed MONAI, using its optimized U-Net is often easier:
# from monai.networks.nets import UNet as MONAI_UNet
#
# def build_monai_unet(n_channels, n_classes):
#     # Example MONAI U-Net configuration (adjust parameters as needed)
#     model = MONAI_UNet(
#         spatial_dims=3, # Assuming 3D data for BraTS
#         in_channels=n_channels,
#         out_channels=n_classes,
#         channels=(16, 32, 64, 128, 256), # Example channel progression
#         strides=(2, 2, 2, 2),
#         num_res_units=2, # Example residual units
#         # norm=Norm.BATCH, # Example normalization
#     )
#     return model