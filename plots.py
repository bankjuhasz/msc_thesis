import torch
from beat_this.inference import load_model

# Define the checkpoint path
checkpoint_path = "checkpoints/S0 shift_tolerant_weighted_bce-h128-augTrueTrueTruect=1 cc=1-v1.ckpt"
#checkpoint_path = "checkpoints/S0 shift_tolerant_weighted_bce-h128-augTrueTrueTruect=1 cc=1.ckpt"

# Load the model using load_model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model(checkpoint_path, device)
model.eval()

# Input tensor
input_tensor = torch.randn(1, 3072, 128, requires_grad=True, device=device)

# Forward pass
output = model(input_tensor)

beat_output = output["beat"]

# Prediction at the center
#center_index = beat_output.shape[1] // 2
chosen_prediction = beat_output[0, -1]

# Grad of the center prediction w.r.t. input
chosen_prediction.backward()

# Gradient for the second half of the input
gradients = input_tensor.grad
#second_half_gradients = gradients[0, center_index+1:]

npgradients = gradients.cpu().numpy()
#npsecondgradients = second_half_gradients.cpu().numpy()

print("Gradients:", gradients)
#print("Second half gradients:", second_half_gradients)