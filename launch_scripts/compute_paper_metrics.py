import argparse
from pathlib import Path

import numpy as np
from pytorch_lightning import Trainer, seed_everything

from beat_this.dataset import BeatDataModule
from beat_this.dataset.dataset import BeatTrackingDataset
from beat_this.inference import load_checkpoint
from beat_this.model.pl_module import PLBeatThis

import wandb
import re

# for repeatability
seed_everything(0, workers=True)


def main(args):
    if len(args.models) == 1:
        print("Single model prediction for", args.models[0])
        # single model prediction
        checkpoint_path = args.models[0]
        checkpoint = load_checkpoint(checkpoint_path)

        # create datamodule
        datamodule = datamodule_setup(checkpoint, args.num_workers, args.datasplit)
        # create model and trainer
        model, trainer = plmodel_setup(
            checkpoint,
            args.eval_trim_beats,
            args.dbn,
            args.gpu,
            segment_metrics=args.segment_metrics,
            causal_inference=args.causal_inference,
        )
        # predict
        metrics, dataset, preds, piece = compute_predictions(model, trainer, datamodule.predict_dataloader())

        # compute averaged metrics
        averaged_metrics = {k: np.mean(v) for k, v in metrics.items()}

        # compute metrics averaged by dataset
        dataset_metrics = {
            k: {d: np.mean(v[dataset == d]) for d in np.unique(dataset)}
            for k, v in metrics.items()
        }

        # print for dataset
        print("Metrics")
        for k, v in averaged_metrics.items():
            print(f"{k}: {v}")
        print("Dataset metrics")
        for k, v in dataset_metrics.items():
            print(k)
            for d, value in v.items():
                print(f"{d}: {value}")
            print("------")

        if args.update_wandb:
            print(f'Attempting to update wandb metrics for run: {args.update_wandb}')
            try:
                # load run
                api = wandb.Api()
                run_path = 'bank_juhasz_msc_thesis/beat_this/' + args.update_wandb
                run = api.run(run_path)

                filtered_metrics = {
                    k: v
                    for k, v in averaged_metrics.items()
                    if re.match(r'segment_\d+_F-measure_(beat|downbeat)$', k)
                }

                # adding datasplit for clarity
                upload_package = {
                    f"{args.datasplit}_{k}": v
                    for k, v in filtered_metrics.items()
                }

                if not upload_package:
                    print("No F-measure metrics found to upload.")
                else:
                    run.summary.update(upload_package)
                    print(f"Uploaded F-measure metrics: {sorted(upload_package.keys())}")

            except Exception as e:
                print(f'wandb metrics update failed due to the following error: {e}')

    else:  # multiple models
        if args.aggregation_type == "mean-std":
            # computing result variability for the same dataset and different model seeds
            # create datamodule only once, as we assume it is the same for all models
            checkpoint = load_checkpoint(args.models[0])
            datamodule = datamodule_setup(checkpoint, args.num_workers, args.datasplit)
            # create model and trainer
            all_metrics = []
            for checkpoint_path in args.models:
                checkpoint = load_checkpoint(checkpoint_path)
                model, trainer = plmodel_setup(
                    checkpoint, args.eval_trim_beats, args.dbn, args.gpu
                )

                metrics, dataset, preds, piece = compute_predictions(
                    model, trainer, datamodule.predict_dataloader()
                )
                # compute averaged metrics for one model
                averaged_metrics = {k: np.mean(v) for k, v in metrics.items()}
                all_metrics.append(averaged_metrics)
            # compute mean and standard deviations for all model averages
            all_metrics_mean = {
                k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]
            }
            all_metrics_std = {
                k: np.std([m[k] for m in all_metrics]) for k in all_metrics[0]
            }
            all_metrics_stats = {
                k: (all_metrics_mean[k], all_metrics_std[k])
                for k, v in all_metrics[0].items()
            }
            # print all metrics
            print("Metrics")
            for k, v in all_metrics_stats.items():
                # round to 3 decimal places
                print(f"{k}: {round(v[0],3)} +- {round(v[1],3)}")
        elif args.aggregation_type == "k-fold":
            # computing results in the K-fold setting. Every fold has a different dataset
            all_piece_metrics = []
            all_piece_dataset = []
            all_piece = []
            # create datamodule for each model
            for i_model, checkpoint_path in enumerate(args.models):
                print(f"Model {i_model+1}/{len(args.models)}")
                checkpoint = load_checkpoint(checkpoint_path)
                datamodule = datamodule_setup(
                    checkpoint, args.num_workers, args.datasplit
                )
                # create model and trainer
                model, trainer = plmodel_setup(
                    checkpoint, args.eval_trim_beats, args.dbn, args.gpu
                )
                # predict
                metrics, dataset, preds, piece = compute_predictions(
                    model, trainer, datamodule.predict_dataloader()
                )
                all_piece_metrics.append(metrics)
                all_piece_dataset.append(dataset)
                all_piece.append(piece)
            # aggregate across folds
            all_piece_metrics = {
                k: np.concatenate([m[k] for m in all_piece_metrics])
                for k in all_piece_metrics[0]
            }
            all_piece_dataset = np.concatenate(all_piece_dataset)
            all_piece = np.concatenate(all_piece)
            # double check that there are no errors in the fold and there are not repeated pieces
            assert len(all_piece) == len(
                np.unique(all_piece)
            ), "There are repeated pieces in the folds"
            dataset_metrics = {
                k: {
                    d: np.mean(v[all_piece_dataset == d])
                    for d in np.unique(all_piece_dataset)
                }
                for k, v in all_piece_metrics.items()
            }
            # print for dataset
            print("Dataset metrics")
            for k, v in dataset_metrics.items():
                print(k)
                for d, value in v.items():
                    print(f"{d}: {round(value,3)}")
                print("------")
        else:
            raise ValueError(f"Unknown aggregation type {args.aggregation_type}")


def datamodule_setup(checkpoint, num_workers, datasplit):
    # Load the datamodule
    print("Creating datamodule")
    data_dir = Path(__file__).parent.parent.relative_to(Path.cwd()) / "data"
    datamodule_hparams = checkpoint["datamodule_hyper_parameters"]
    # update the hparams with the ones from the arguments
    if num_workers is not None:
        datamodule_hparams["num_workers"] = num_workers
    datamodule_hparams["predict_datasplit"] = datasplit
    datamodule_hparams["data_dir"] = data_dir

    datamodule = BeatDataModule(**datamodule_hparams)
    datamodule.setup(stage="predict")

    return datamodule


def plmodel_setup(
        checkpoint,
        eval_trim_beats,
        dbn,
        gpu,
        segment_metrics=False,
        causal_inference=False,
):
    """
    Set up the pytorch lightning model and trainer for evaluation.

    Args:
        checkpoint (dict): The dict containing the checkpoint to load.
        eval_trim_beats (int or None): The number of beats to trim during evaluation. If None, the setting is taken from the pretrained model.
        dbn (bool or None): Whether to use the Dynamic Bayesian Network (DBN) module during evaluation. If None, the default behavior from the pretrained model is used.
        gpu (int): The index of the GPU device to use for training.
        segment_metrics (bool): Whether to compute metrics in 10s segments per excerpt.
        causal_inference (bool): Whether to compute metrics without stitching and to use causal postp.

    Returns:
        tuple: A tuple containing the initialized pytorch lightning model and trainer.

    """
    if eval_trim_beats is not None:
        checkpoint["hyper_parameters"]["eval_trim_beats"] = eval_trim_beats
    if dbn is not None:
        checkpoint["hyper_parameters"]["use_dbn"] = dbn
    checkpoint["hyper_parameters"]["segment_metrics"] = segment_metrics
    checkpoint["hyper_parameters"]["causal_inference"] = causal_inference

    model = PLBeatThis(**checkpoint["hyper_parameters"])
    model.load_state_dict(checkpoint["state_dict"])
    # set correct device and accelerator
    if gpu >= 0:
        devices = [gpu]
        accelerator = "gpu"
    else:
        devices = 1
        accelerator = "cpu"
    # create trainer
    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=None,
        deterministic=True,
        precision="16-mixed",
    )
    return model, trainer


def compute_predictions(model, trainer, predict_dataloader):
    print("Computing predictions ...")
    out = trainer.predict(model, predict_dataloader)

    # during segmented evaluation, it can happen that no metrics are calculated due to the excerpt being too short
    # in these cases, empty dicts are returned for metrics. we therefore exclude these pieces entirely and remove them
    # from the returned lists.
    filtered = [
        (m, p, d, pc)
        for (m, p, d, pc) in out
        if m  # keeps only truthy m, i.e. non‐empty dict
    ]

    if filtered:
        metrics, preds, dataset, piece = zip(*filtered)
        metrics = list(metrics)
        preds = list(preds)
        dataset = np.asarray([dlist[0] for dlist in dataset])
        piece = np.asarray([pclist[0] for pclist in piece])
    else:
        raise Exception("No valid metrics found.")

    # convert metrics from list of per-batch dictionaries to a single dictionary with np arrays as values
    metrics = {k: np.asarray([m[k] for m in metrics]) for k in metrics[0]}

    return metrics, dataset, preds, piece


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computes predictions for a given model and dataset, "
        "prints metrics, and optionally dumps predictions to a given file."
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="Local checkpoint files to use",
    )
    parser.add_argument(
        "--datasplit",
        type=str,
        choices=("train", "val", "test"),
        default="val",
        help="data split to use: train, val or test " "(default: %(default)s)",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--num_workers", type=int, default=8, help="number of data loading workers "
    )
    parser.add_argument(
        "--eval_trim_beats",
        metavar="SECONDS",
        type=float,
        default=None,
        help="Override whether to skip the first given seconds "
        "per piece in evaluating (default: as stored in model)",
    )
    parser.add_argument(
        "--dbn",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="override the option to use madmom postprocessing dbn",
    )
    parser.add_argument(
        "--aggregation-type",
        type=str,
        choices=("mean-std", "k-fold"),
        default="mean-std",
        help="Type of aggregation to use for multiple models; ignored if only one model is given",
    )
    parser.add_argument(
        "--segment_metrics",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If True, compute metrics separately for 3 10s segments within each 30s excerpt."
    )
    parser.add_argument(
        "--update_wandb",
        type=str,
        default=None,
        help="W&B run ID of the run which the evaluation metrics will be uploaded to."
    )
    parser.add_argument(
        "--causal_inference",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="If True, compute metrics for the full piece instead of segments or stitched-together excerpts of size chunk_size and apply causal postprocessing."
    )

    args = parser.parse_args()

    main(args)
