import torch

# Load the original tensor
train_ids = torch.load("Transformer/Dataset/train_ids.pt", weights_only=True)

# Keep only the first 1000 tokens
small_train_ids = train_ids[:1000].clone()

# Save it
torch.save(small_train_ids, "Transformer/Dataset/train_ids_1000.pt")

print(f"Original tokens: {len(train_ids)}")
print(f"New tokens: {len(small_train_ids)}")