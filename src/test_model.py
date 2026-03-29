import torch
from model import UNet

# Create model
model = UNet()

# Dummy input (batch_size=1)
before = torch.randn(1, 3, 256, 256)
after  = torch.randn(1, 3, 256, 256)

# Forward pass
output = model(before, after)

print("Output shape:", output.shape)
