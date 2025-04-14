import os

import torch
import torch.nn as nn

# Import the modified ResNet classes from the downloaded resnet.py file
from resnet import ResNet50  # Use the ResNet50 class defined in resnet.py


class SimCLRModel(nn.Module):
    def __init__(
        self,
        base_model='resnet50',
        projection_dim=128,
        checkpoint_path="models/resnet_pretrained_models/resnet50_cifar10_bs1024_epochs1000.pth.tar"
    ):
        super().__init__()
        self.checkpoint_path = checkpoint_path

        # Load the modified ResNet50 model (with CIFAR-10 compatible architecture)
        resnet = ResNet50(cifar_head=True)  # Use cifar_head=True for 3x3 conv
        feat_dim = resnet.fc.in_features if hasattr(resnet, 'fc') else 2048  # Default feature dim for ResNet50

        # Remove the final fc layer and replace it with Identity
        resnet.fc = nn.Identity()
        self.convnet = resnet

        # Define the projection head to match the checkpoint's structure
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, projection_dim, bias=False),
            nn.BatchNorm1d(projection_dim, affine=False)
        )

        # Rename layers in the projection head to match checkpoint keys
        self.projection[0].name = 'fc1'
        self.projection[1].name = 'bn1'
        self.projection[3].name = 'fc2'
        self.projection[4].name = 'bn2'

        # Load checkpoint if it exists
        if self.checkpoint_path is not None and os.path.isfile(self.checkpoint_path):
            self.load_checkpoint()
        else:
            print(f"[Warning] No checkpoint found at: {self.checkpoint_path}")

    def forward(self, x):
        features = self.convnet(x)
        projections = self.projection(features)
        return projections

    def load_checkpoint(self):
        print(f"[Info] Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        # Some checkpoints wrap model in 'state_dict'
        state_dict = checkpoint.get('state_dict', checkpoint)

        # Adjust state_dict keys to match the current model's structure
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]  # Remove "module." prefix
            # Map checkpoint keys to model keys for the projection head
            if k.startswith("projection."):
                parts = k.split('.')
                if parts[1] == 'fc1':
                    parts[1] = '0'
                elif parts[1] == 'bn1':
                    parts[1] = '1'
                elif parts[1] == 'fc2':
                    parts[1] = '3'
                elif parts[1] == 'bn2':
                    parts[1] = '4'
                k = '.'.join(parts)
            new_state_dict[k] = v

        # Load the adjusted state_dict into the model
        self.load_state_dict(new_state_dict, strict=False)
        print("[Info] Checkpoint loaded successfully.")


# Instantiate the model
model = SimCLRModel()
model.eval()

# Test the model with random input
x = torch.randn(2, 3, 32, 32)  # CIFAR-10 images are typically 32x32
with torch.no_grad():
    z = model(x)
print(z.shape)  # Should be [2, 128]
print(z)