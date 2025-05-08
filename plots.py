import torch
import numpy as np
import matplotlib.pyplot as plt
from beat_this.inference import load_model

def plot_gradients_over_time(
        checkpoint_path = str,
        input_tensor = torch.Tensor,
        chosen_frame = int,
        custom_model_name = str,
        target = str,
):
    '''
    Shows the gradients of the chosen frame prediction (beats) w.r.t. the input tensor.

    Args:
        checkpoint_path: relative path to checkpoint file to evaluate
        input_tensor: shape (1, sequence length, 128), requires_grad=True
        chosen_frame: index of the frame to evaluate
        custom_model_name: custom name for the loaded model to show on the plot
        target: either "model_output" or "frontend_output"
    '''

    # load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(checkpoint_path, device)
    model.eval()

    if target == "model_output":
        # forward pass
        output = model(input_tensor)
        beat_output = output["beat"]

        # grad of the frame prediction w.r.t. input
        chosen_prediction = beat_output[0, chosen_frame].sum()
        chosen_prediction.backward()

    elif target == "frontend_output":
        x_feat = model.frontend(input_tensor)
        x_feat.retain_grad()

        feat_scalar = x_feat[0, chosen_frame].sum()
        feat_scalar.backward()

    else:
        raise ValueError("target must be either 'model_output' or 'frontend_output'")

    gradients = input_tensor.grad

    # sum grads across the frequency axis
    summed_gradients = gradients[0].abs().sum(dim=1).cpu().numpy()

    # find the first idx where the gradient is above 0
    first_positive_index = np.argmax(summed_gradients > 0)
    first_positive_value = summed_gradients[first_positive_index]

    # last idx where the gradient is above 0
    last_positive_index = summed_gradients.shape[0] - 1 - np.argmax((summed_gradients[::-1] > 0))

    # plot the summed gradients
    plt.figure(figsize=(10, 6))
    plt.plot(summed_gradients, label="Summed Gradients (Abs. Value)")
    plt.axvline(x=first_positive_index, color='red', linestyle='--', label=f"First > 0: {first_positive_index}")
    plt.text(first_positive_index, first_positive_value, f"{first_positive_value:.8f}", color='red', fontsize=10, ha='right')
    plt.axvline(last_positive_index, color='green', linestyle='--', label=f"Last  > 0: {last_positive_index}")
    plt.xlabel("Time Steps")
    plt.ylabel("Gradient Magnitude")
    if custom_model_name:
        plt.title(f"Input: input_tensor | Model: {custom_model_name} | Frame: {chosen_frame}")
    else:
        plt.title("Summed Gradients Over Time (Abs. Value)")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    torch.manual_seed(0)
    #checkpoint_path = "small0"
    #checkpoint_path = "checkpoints/S0 shift_tolerant_weighted_bce-h128-augTrueTrueTruect=1.ckpt"
    checkpoint_path = "checkpoints/S0 shift_tolerant_weighted_bce-h128-augTrueTrueTruect=1 cc=1.ckpt"
    #checkpoint_path = "checkpoints/S0 shift_tolerant_weighted_bce-h128-augTrueTrueTruect=1 cc=1-v1.ckpt"

    input_tensor = torch.randn(1, 1500, 128, requires_grad=True, device='cuda')
    chosen_frame = 750

    #from beat_this.preprocessing import load_audio
    #real_audio, sr = load_audio("tests/It Don't Mean A Thing - Kings of Swing.mp3")
    #print("real_audio shape:", real_audio.shape)
    #print("real_audio type:", type(real_audio))

    #custom_model_name = "small0"
    #custom_model_name = "causal_trans + normal_conv"
    custom_model_name = "causal_trans + causal_conv"
    #custom_model_name = "sliding window attention"

    plot_gradients_over_time(checkpoint_path, input_tensor, chosen_frame, custom_model_name, target="model_output")
