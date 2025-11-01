from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


class Postprocessor:
    """Postprocessor for the beat and downbeat predictions of the model.
    The postprocessor takes the (framewise) model predictions (beat and downbeats) and the padding mask,
    and returns the postprocessed beat and downbeat as list of times in seconds.
    The beats and downbeats can be 1D arrays (for only 1 piece) or 2D arrays, if a batch of pieces is considered.
    The output dimensionality is the same as the input dimensionality.
    Two types of postprocessing are implemented:
        - minimal: a simple postprocessing that takes the maximum of the framewise predictions,
        and removes adjacent peaks.
        - dbn: a postprocessing based on the Dynamic Bayesian Network proposed by Böck et al.
    Args:
        type (str): the type of postprocessing to apply. Either "minimal" or "dbn". Default is "minimal".
        fps (int): the frames per second of the model framewise predictions. Default is 50.
    """

    def __init__(self, type: str = "minimal", fps: int = 50):
        assert type in ["minimal", "dbn", "no_postprocessing", "causal_thresholding", "shifted_causal_local_max"]
        self.type = type
        self.fps = fps
        if type == "dbn":
            from madmom.features.downbeats import DBNDownBeatTrackingProcessor

            self.dbn = DBNDownBeatTrackingProcessor(
                beats_per_bar=[3, 4],
                min_bpm=55.0,
                max_bpm=215.0,
                fps=self.fps,
                transition_lambda=100,
            )

    def __call__(
        self,
        beat: torch.Tensor,
        downbeat: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        shift: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply postprocessing to the input beat and downbeat tensors. Works with batched and unbatched inputs.
        The output is a list of times in seconds, or a list of lists of times in seconds, if the input is batched.

        Args:
            beat (torch.Tensor): The input beat tensor.
            downbeat (torch.Tensor): The input downbeat tensor.
            padding_mask (torch.Tensor, optional): The padding mask tensor. Defaults to None.

        Returns:
            torch.Tensor: The postprocessed beat tensor.
            torch.Tensor: The postprocessed downbeat tensor.
        """
        batched = False if beat.ndim == 1 else True
        if padding_mask is None:
            padding_mask = torch.ones_like(beat, dtype=torch.bool)

        # if beat and downbeat are 1D tensors, add a batch dimension
        if not batched:
            beat = beat.unsqueeze(0)
            downbeat = downbeat.unsqueeze(0)
            padding_mask = padding_mask.unsqueeze(0)

        if self.type == "minimal":
            postp_beat, postp_downbeat = self.postp_minimal(
                beat, downbeat, padding_mask
            )
        elif self.type == "dbn":
            postp_beat, postp_downbeat = self.postp_dbn(beat, downbeat, padding_mask)
        elif self.type == "no_postprocessing":
            postp_beat, postp_downbeat = self.no_postprocessing(beat, downbeat, padding_mask)
        elif self.type == "causal_thresholding":
            postp_beat, postp_downbeat = self.causal_thresholding(beat, downbeat)
        elif self.type == "shifted_causal_local_max":
            postp_beat, postp_downbeat = self.shifted_causal_local_max(beat, downbeat, shift=shift)
        else:
            raise ValueError("Invalid postprocessing type")

        # remove the batch dimension if it was added
        if not batched:
            postp_beat = postp_beat[0]
            postp_downbeat = postp_downbeat[0]

        # update the model prediction dict
        return postp_beat, postp_downbeat

    def postp_minimal(self, beat, downbeat, padding_mask):
        # concatenate beat and downbeat in the same tensor of shape (B, T, 2)
        packed_pred = rearrange(
            [beat, downbeat], "c b t -> b t c", b=beat.shape[0], t=beat.shape[1], c=2
        )
        # set padded elements to -1000 (= probability zero even in float64) so they don't influence the maxpool
        pred_logits = packed_pred.masked_fill(~padding_mask.unsqueeze(-1), -1000)
        # reshape to (2*B, T) to apply max pooling
        pred_logits = rearrange(pred_logits, "b t c -> (c b) t")
        # pick maxima within +/- 70ms
        pred_peaks = pred_logits.masked_fill(
            pred_logits != F.max_pool1d(pred_logits, 7, 1, 3), -1000
        )
        # keep maxima with over 0.5 probability (logit > 0)
        pred_peaks = pred_peaks > 0
        #  rearrange back to two tensors of shape (B, T)
        beat_peaks, downbeat_peaks = rearrange(
            pred_peaks, "(c b) t -> c b t", b=beat.shape[0], t=beat.shape[1], c=2
        )
        # run the piecewise operations
        with ThreadPoolExecutor() as executor:
            postp_beat, postp_downbeat = zip(
                *executor.map(
                    self._postp_minimal_item, beat_peaks, downbeat_peaks, padding_mask
                )
            )
        return postp_beat, postp_downbeat

    def _postp_minimal_item(self, padded_beat_peaks, padded_downbeat_peaks, mask):
        """Function to compute the operations that must be computed piece by piece, and cannot be done in batch."""
        # unpad the predictions by truncating the padding positions
        beat_peaks = padded_beat_peaks[mask]
        downbeat_peaks = padded_downbeat_peaks[mask]
        # pass from a boolean array to a list of times in frames.
        beat_frame = torch.nonzero(beat_peaks).cpu().numpy()[:, 0]
        downbeat_frame = torch.nonzero(downbeat_peaks).cpu().numpy()[:, 0]
        # remove adjacent peaks
        beat_frame = deduplicate_peaks(beat_frame, width=1)
        downbeat_frame = deduplicate_peaks(downbeat_frame, width=1)
        # convert from frame to seconds
        beat_time = beat_frame / self.fps
        downbeat_time = downbeat_frame / self.fps
        # move the downbeat to the nearest beat
        if (
            len(beat_time) > 0
        ):  # skip if there are no beats, like in the first training steps
            for i, d_time in enumerate(downbeat_time):
                beat_idx = np.argmin(np.abs(beat_time - d_time))
                downbeat_time[i] = beat_time[beat_idx]
        # remove duplicate downbeat times (if some db were moved to the same position)
        downbeat_time = np.unique(downbeat_time)
        return beat_time, downbeat_time

    def postp_dbn(self, beat, downbeat, padding_mask):
        beat_prob = beat.double().sigmoid()
        downbeat_prob = downbeat.double().sigmoid()
        # limit lower and upper bound, since 0 and 1 create problems in the DBN
        epsilon = 1e-5
        beat_prob = beat_prob * (1 - epsilon) + epsilon / 2
        downbeat_prob = downbeat_prob * (1 - epsilon) + epsilon / 2
        with ThreadPoolExecutor() as executor:
            postp_beat, postp_downbeat = zip(
                *executor.map(
                    self._postp_dbn_item, beat_prob, downbeat_prob, padding_mask
                )
            )
        return postp_beat, postp_downbeat

    def _postp_dbn_item(self, padded_beat_prob, padded_downbeat_prob, mask):
        """Function to compute the operations that must be computed piece by piece, and cannot be done in batch."""
        # unpad the predictions by truncating the padding positions
        beat_prob = padded_beat_prob[mask]
        downbeat_prob = padded_downbeat_prob[mask]
        # build an artificial multiclass prediction, as suggested by Böck et al.
        # again we limit the lower bound to avoid problems with the DBN
        epsilon = 1e-5
        combined_act = np.vstack(
            (
                np.maximum(
                    beat_prob.cpu().numpy() - downbeat_prob.cpu().numpy(), epsilon / 2
                ),
                downbeat_prob.cpu().numpy(),
            )
        ).T
        # run the DBN
        dbn_out = self.dbn(combined_act)
        postp_beat = dbn_out[:, 0]
        postp_downbeat = dbn_out[dbn_out[:, 1] == 1][:, 0]
        return postp_beat, postp_downbeat

    def no_postprocessing(self, beat, downbeat, padding_mask):
        """ Does no real postprocessing, just returns the times, where the beat/downbeat prediction is over 0.5 """
        # Ensure batched shape
        if beat.ndim == 1:
            beat = beat.unsqueeze(0)
            downbeat = downbeat.unsqueeze(0)
            padding_mask = padding_mask.unsqueeze(0)
        batch_size = beat.shape[0]
        postp_beat = []
        postp_downbeat = []
        for i in range(batch_size):
            mask = padding_mask[i]
            beat_frames = torch.nonzero((beat[i] > 0.5) & mask).cpu().numpy()[:, 0]
            downbeat_frames = torch.nonzero((downbeat[i] > 0.5) & mask).cpu().numpy()[:, 0]
            postp_beat.append(beat_frames / self.fps)
            postp_downbeat.append(downbeat_frames / self.fps)
        return tuple(postp_beat), tuple(postp_downbeat)

    def causal_thresholding(self, beat, downbeat, threshold=1.75, cooldown=5):
        """ Simple causal thresholding: a beat/downbeat is detected if the activation is above threshold, but there is
        a cooldown period (measured in FRAMES) after each detection, during which no new beat/downbeat can be detected."""
        # Ensure batched shape
        if beat.ndim == 1:
            beat = beat.unsqueeze(0)
            downbeat = downbeat.unsqueeze(0)

        B, T = beat.shape[0], beat.shape[1]
        fps = float(self.fps)
        db_cooldown = 2 * int(cooldown)
        db_threshold = threshold * 0.4

        postp_beat, postp_downbeat = [], []

        for i in range(B):
            beat_times, down_times = [], []
            last_beat = -10 ** 9
            last_down = -10 ** 9

            for t in range(T):
                b = float(beat[i, t])
                d = float(downbeat[i, t])

                # beat decision first --> only allow downbeat if there is a beat
                if b > threshold and (t - last_beat) >= cooldown:
                    t_sec = t / fps
                    beat_times.append(t_sec)
                    last_beat = t

                    if d > db_threshold and (t - last_down) >= db_cooldown:
                        down_times.append(t_sec)
                        last_down = t

            postp_beat.append(np.array(beat_times, dtype=np.float32))
            postp_downbeat.append(np.array(down_times, dtype=np.float32))

        return tuple(postp_beat), tuple(postp_downbeat)

    def shifted_causal_local_max(self, beat, downbeat, threshold=1, shift=3, cooldown=5, db_anywhere=True):
        """ Causal local max detection with a shift/lookahead of `shift` frames. Makes use of the fact that shifted
        models look `shift` frames into the future in order to identify local maxima more reliably."""

        # Ensure batched shape
        if beat.ndim == 1:
            beat = beat.unsqueeze(0)
            downbeat = downbeat.unsqueeze(0)

        B, T = beat.shape[0], beat.shape[1]
        fps = float(self.fps)
        db_cooldown = 2 * int(cooldown)

        postp_beat, postp_down = [], []
        win_len = 2 * shift + 1
        center_idx = shift

        ### TEST
        db_threshold = threshold * 1 #.5

        for i in range(B):
            beat_times, down_times = [], []
            last_beat = -10 ** 9
            last_down = -10 ** 9

            # sliding windows for beats and downbeats
            winB: list[float] = []
            winD: list[float] = []

            for t in range(T):
                # append current votes
                winB.append(float(beat[i, t]))
                winD.append(float(downbeat[i, t]))

                # keep only the last 2*shift+1 entries
                if len(winB) > win_len:
                    winB.pop(0)
                    winD.pop(0)

                # need a full symmetric window to decide
                if len(winB) < win_len:
                    continue

                # Center-only local max for BEAT at τ = t
                cB = winB[center_idx]
                leftB = winB[:center_idx]
                rightB = winB[center_idx + 1:]
                neigh_max_B = max(leftB + rightB) if (leftB or rightB) else float("-inf")

                if cB >= threshold and cB >= neigh_max_B and (t - last_beat) >= cooldown:
                    # only emit beat if: above threshold, is a local max in the center of the window, and cooldown passed
                    time_s = t / fps
                    beat_times.append(time_s)
                    last_beat = t

                    if db_anywhere:
                        # downbeats --> can be anywhere in the window
                        maxD = max(winD)
                        if maxD >= db_threshold and (t - last_down) >= db_cooldown:
                            down_times.append(time_s)
                            last_down = t

                    else:
                        # downbeats --> same logic as beats, but only if there was a beat
                        cD = winD[center_idx]
                        leftD = winD[:center_idx]
                        rightD = winD[center_idx + 1:]
                        neigh_max_D = max(leftD + rightD) if (leftD or rightD) else float("-inf")

                        if cD >= threshold and cD >= neigh_max_D and (t - last_down) >= db_cooldown:
                            down_times.append(time_s)
                            last_down = t

            postp_beat.append(np.array(beat_times, dtype=np.float32))
            postp_down.append(np.array(down_times, dtype=np.float32))

        return tuple(postp_beat), tuple(postp_down)


def deduplicate_peaks(peaks, width=1) -> np.ndarray:
    """
    Replaces groups of adjacent peak frame indices that are each not more
    than `width` frames apart by the average of the frame indices.
    """
    result = []
    peaks = map(int, peaks)  # ensure we get ordinary Python int objects
    try:
        p = next(peaks)
    except StopIteration:
        return np.array(result)
    c = 1
    for p2 in peaks:
        if p2 - p <= width:
            c += 1
            p += (p2 - p) / c  # update mean
        else:
            result.append(p)
            p = p2
            c = 1
    result.append(p)
    return np.array(result)
