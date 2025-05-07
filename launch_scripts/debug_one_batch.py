import torch

# Load the checkpoint. Use map_location='cpu' to load on CPU if needed.
checkpoint = torch.load('checkpoints/S0 noval shift_tolerant_weighted_bce-h128-augTrueTrueTrue_CAUSAL_.ckpt', map_location='cpu')

# Extract the state dictionary
state_dict = checkpoint.get('state_dict', checkpoint)

# Print out the keys to see what parameters are saved.
print("State dict keys:")
for key in state_dict.keys():
    print(key)