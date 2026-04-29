import torch

# Load checkpoint
checkpoint = torch.load('checkpoints/improved_residual_best.pt', map_location='cpu')

# Extract model state dict
model_state_dict = checkpoint['model_state_dict']

# Save as .net (Meta format - just the state dict)
torch.save(model_state_dict, 'checkpoints/improved_residual_best.net')

print("✅ Converted to .net format")
print(f"Epoch: {checkpoint['epoch']}")
print(f"Val Loss: {checkpoint['val_loss']:.6f}")
print(f"Val Metrics: {checkpoint['val_metrics']}")
