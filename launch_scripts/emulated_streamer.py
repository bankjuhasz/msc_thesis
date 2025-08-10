import argparse
from pathlib import Path
import torch

from beat_this.dataset.dataset import BeatDataModule
from beat_this.inference import streaming_predict, load_model

def main(args):
    assert args.device in ["cpu", "gpu"]

    data_dir = Path(__file__).parent.parent.relative_to(Path.cwd()) / "data"
    dm = BeatDataModule(
        data_dir=data_dir,
        spect_fps=50,
        test_dataset="gtzan",
        fold=None,
        predict_datasplit="val", # we're intending to predict on the full songs in the validation set
    )
    dm.setup("predict")
    pred_loader = dm.predict_dataloader()

    model = load_model(args.model_ckpt)

    for batch in pred_loader:
        spect = batch["spect"].unsqueeze(0)  # (1, T, F)
        out = streaming_predict(
            model,
            spect=spect,
            window_size=args.window_size,
            peek_size=args.peek_size,
            device=args.device,
            tolerance=3,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Emulates real-time audio streaming for the purpose of latency testing. Expects precomputed spectrograms"
                    "as we're only interested in the latency of the model inference, not the audio processing."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices=["cpu", "gpu"],
        help="Device to run the model on (default: gpu). If set to 'gpu', it will use the first available GPU, otherwise it will use CPU."
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=512,
        help="Size of the audio window (in frames) to process at a time (default: 512)."
    )
    parser.add_argument(
        "--peek-size",
        type=int,
        default=2,
        help="Number of frames to peek ahead in the audio stream, i.e., nr of new frames per iteration (default: 2)."
    )
    parser.add_argument(
        "--cache-conv",
        type=bool,
        default=False,
        help="Whether to cache convolutional layers for faster inference (default: False)."
    )
    parser.add_argument(
        "--cache-kv",
        type=bool,
        default=False,
        help="Whether to cache keys and values for attention for faster inference (default: False)."
    )
    parser.add_argument(
        "--model-ckpt",
        type=str,
        required=True,
        help="Path to the model checkpoint file."
    )

    args = parser.parse_args()

    main(args)