#ResUNet Architecture for Kidney Stone Segmentation

import torch
import torch.nn as nn
from torchgen import model
import torchvision

class ResUNet(nn.Module):       
    def __init__(self):
        #Initialize the ResUNet architecture.
        super().__init__()
        resnet = torchvision.models.resnet50(weights="DEFAULT") # LOAD PRETRAINED RESNET50
        # MODIFY FIRST CONV LAYER FOR GRAYSCALE INPUT
        resnet.conv1 = nn.Conv2d(
            in_channels=1,      # Grayscale input
            out_channels=64,    # Same as original
            kernel_size=7,      # Same as original
            stride=2,           # Same as original
            padding=3,          # Same as original
            bias=False          # Same as original
        )

        # ENCODER (DOWN-SAMPLING PATH)
        self.enc1 = nn.Sequential(
            resnet.conv1,   # Conv2d(1, 64, 7, stride=2)
            resnet.bn1,     # BatchNorm2d(64)
            resnet.relu     # ReLU activation
        )
        
        # Max pooling after enc1
        # Input: (64, 128, 128) → Output: (64, 64, 64)
        self.pool = resnet.maxpool
        # Input: (64, 64, 64) → Output: (256, 64, 64)
        self.enc2 = resnet.layer1
        # Input: (256, 64, 64) → Output: (512, 32, 32)
        self.enc3 = resnet.layer2
        # Input: (512, 32, 32) → Output: (1024, 16, 16)
        self.enc4 = resnet.layer3
        # Input: (1024, 16, 16) → Output: (2048, 8, 8)
        self.enc5 = resnet.layer4
        
        # DECODER (UP-SAMPLING PATH WITH SKIP CONNECTIONS)
        self.up1 = nn.ConvTranspose2d(
            in_channels=2048,
            out_channels=1024,
            kernel_size=2,
            stride=2
        )
        # conv1: 1024 (from up1) + 1024 (from enc4) = 2048 → 512
        self.conv1 = nn.Conv2d(
            in_channels=2048,  # 1024 + 1024
            out_channels=512,
            kernel_size=3,
            padding=1
        )
        
        # -------- Up 2: 512 → 256 → 256 --------
        # Up-sample from 16x16 to 32x32
        # Concatenate with enc3 (skip connection)
        # Reduce to 256 channels
        self.up2 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=2,
            stride=2
        )
        # conv2: 256 (from up2) + 512 (from enc3) = 768 → 256
        self.conv2 = nn.Conv2d(
            in_channels=768,   # 256 + 512
            out_channels=256,
            kernel_size=3,
            padding=1
        )
        
        # -------- Up 3: 256 → 128 → 128 --------
        # Up-sample from 32x32 to 64x64
        # Concatenate with enc2 (skip connection)
        # Reduce to 128 channels
        self.up3 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=2,
            stride=2
        )
        # conv3: 128 (from up3) + 256 (from enc2) = 384 → 128
        self.conv3 = nn.Conv2d(
            in_channels=384,   # 128 + 256
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        
        # -------- Up 4: 128 → 64 → 64 --------
        # Up-sample from 64x64 to 128x128
        # Concatenate with enc1 (skip connection)
        # Reduce to 64 channels
        self.up4 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2
        )
        # conv4: 64 (from up4) + 64 (from enc1) = 128 → 64
        self.conv4 = nn.Conv2d(
            in_channels=128,   # 64 + 64
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        
        # -------- Up 5: 64 → 32 --------
        # Final up-sampling to restore original size
        # Up-sample from 128x128 to 256x256
        self.up5 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=2,
            stride=2
        )
        
        # -------- Final Convolution: 32 → 1 --------
        # Reduce to single channel (binary segmentation)
        # Use 1x1 convolution to combine features
        self.final = nn.Conv2d(
            in_channels=32,
            out_channels=1,
            kernel_size=1
        )
        
        # INITIALIZE WEIGHTS
        for m in [self.up1, self.up2, self.up3, self.up4, self.up5,
                  self.conv1, self.conv2, self.conv3, self.conv4, self.final]:
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Forward pass through the network. 
        
        # ENCODER PATH
        # Save skip connections at each stage for the decoder
        
        # Stage 1: Initial convolution
        # x1 shape: (batch, 64, 128, 128)
        x1 = self.enc1(x)  # Conv + BN + ReLU
        
        # Stage 2: After maxpool + layer1 
        x2 = self.pool(x1)# x2 shape: (batch, 256, 64, 64)
        x2 = self.enc2(x2)
        
        # Stage 3: After layer2
        x3 = self.enc3(x2)# x3 shape: (batch, 512, 32, 32)
        
        # Stage 4: After layer3
        x4 = self.enc4(x3)# x4 shape: (batch, 1024, 16, 16)
        
        # Stage 5: After layer4 (bottleneck)
        x5 = self.enc5(x4)# x5 shape: (batch, 2048, 8, 8)
        
        # DECODER PATH WITH SKIP CONNECTIONS
        #Decoder Stage 1
        # Up-sample: 8x8 → 16x16
        x = self.up1(x5)  # (B, 1024, 16, 16)
        
        # Concatenate with skip connection from enc4
        # This combines high-level features with spatial details
        x = torch.cat([x, x4], dim=1)  # (B, 2048, 16, 16)
        
        # Reduce channels
        x = torch.relu(self.conv1(x))  # (B, 512, 16, 16)
        
        #Decoder Stage 2
        # Up-sample: 16x16 → 32x32
        x = self.up2(x)  # (B, 256, 32, 32)
        
        # Concatenate with skip connection from enc3
        x = torch.cat([x, x3], dim=1)  # (B, 768, 32, 32)
        
        # Reduce channels
        x = torch.relu(self.conv2(x))  # (B, 256, 32, 32)
        
        # Decoder Stage 3
        # Up-sample: 32x32 → 64x64
        x = self.up3(x)  # (B, 128, 64, 64)
        
        # Concatenate with skip connection from enc2
        x = torch.cat([x, x2], dim=1)  # (B, 384, 64, 64)
        
        # Reduce channels
        x = torch.relu(self.conv3(x))  # (B, 128, 64, 64)
        
        # Decoder Stage 4
        # Up-sample: 64x64 → 128x128
        x = self.up4(x)  # (B, 64, 128, 128)
        
        # Concatenate with skip connection from enc1
        x = torch.cat([x, x1], dim=1)  # (B, 128, 128, 128)
        
        # Reduce channels
        x = torch.relu(self.conv4(x))  # (B, 64, 128, 128)
        
        # Decoder Stage 5 (Final)
        # Up-sample: 128x128 → 256x256
        x = self.up5(x)  # (B, 32, 256, 256)
        
        # Final 1x1 convolution to get single channel
        x = self.final(x)  # (B, 1, 256, 256)
        

        # APPLY SIGMOID ACTIVATION
        return torch.sigmoid(x)
