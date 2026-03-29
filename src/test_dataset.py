from dataset import LEVIRDataset
from torch.utils.data import DataLoader

# Create dataset
train_dataset = LEVIRDataset("../data/train")

# Create dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)

# Test one batch
before, after, label = next(iter(train_loader))

print("Before batch shape:", before.shape)
print("After batch shape:", after.shape)
print("Label batch shape:", label.shape)
