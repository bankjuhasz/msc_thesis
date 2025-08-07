import sounddevice as sd
import soundfile as sf
import numpy as np
import torch
import torchaudio
import threading
import queue

from beat_this.preprocessing import LogMelSpect
from beat_this.inference import load_model, load_audio

def record_and_save(filename, duration=5, samplerate=16000, channels=1):
    """ Record `duration` seconds from the default microphone and write to `filename` (WAV). """
    print(f"Recording {duration} seconds at {samplerate}Hz...")
    data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='float32')
    sd.wait()  # block until recording is finished
    sf.write(filename, data, samplerate)
    print(f"Saved recording to {filename!r}")

def prepare_audio_file(waveform, orig_sr, target_sr=22050):
    """ Resample audio to target sample rate and convert to mono if needed. """
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.from_numpy(waveform)
    if orig_sr != target_sr:
        waveform = torchaudio.transforms.Resample(orig_sr, target_sr)(waveform.T)
    if waveform.ndim > 1:
        waveform = torch.mean(waveform, dim=0)  # convert to mono
    return waveform.numpy(), target_sr

def make_tone(freq: float, duration: float, sample_rate: int = 44100) -> np.ndarray:
    """ Generate a sine wave tone. """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone = np.sin(2 * np.pi * freq * t)
    return tone

class StreamProcessor:
    def __init__(
        self,
        model_ckpt: str,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 441,
        max_context: int = 500,
    ):
        # Load and set up the model
        self.model = load_model(model_ckpt)
        self.model.eval()

        # Preprocessor for log-mel spectrograms
        self.preproc = LogMelSpect(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            device="cpu",
        )

        # Buffers for raw audio and spectrogram frames
        self.raw_buffer = np.zeros(0, dtype=np.float32)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_context = max_context

        # Spectrogram buffer: (time_frames, n_mels)
        self.spec_buffer = np.zeros((0, self.preproc.spect_class.n_mels), dtype=np.float32)
        self._prev_frames = 0

    def process_chunk(self, chunk: np.ndarray) -> (bool, bool):
        """
        Append a chunk of raw audio and produce (beat_flag, downbeat_flag).
        """
        beat_flag = False
        downbeat_flag = False

        # Append new samples
        self.raw_buffer = np.concatenate([self.raw_buffer, chunk])

        # Process as long as we can compute at least one FFT window
        while len(self.raw_buffer) >= self.n_fft:
            frame = self.raw_buffer[:self.n_fft]
            x = self.preproc.device  # ensure CPU tensor
            mel_spec = self.preproc(torch.from_numpy(frame).to(self.preproc.device))  # (n_frames, n_mels)

            total_frames = mel_spec.shape[0]
            new_frames = mel_spec[self._prev_frames : total_frames]
            self._prev_frames = total_frames

            if new_frames.shape[0] > 0:
                # Append and crop context
                self.spec_buffer = np.vstack([self.spec_buffer, new_frames.cpu().numpy()])
                if self.spec_buffer.shape[0] > self.max_context:
                    self.spec_buffer = self.spec_buffer[-self.max_context :]

                # Model inference
                with torch.no_grad():
                    inp = torch.from_numpy(self.spec_buffer).unsqueeze(0)  # (1, T, F)
                    outputs = self.model(inp)
                    beat_logit = outputs['beat'][0, -1].cpu().item()
                    down_logit = outputs['downbeat'][0, -1].cpu().item()

                beat_flag = beat_logit > 0.5
                downbeat_flag = down_logit > 0.5

            # Slide window
            self.raw_buffer = self.raw_buffer[self.hop_length :]

        print(beat_flag, downbeat_flag)
        return beat_flag, downbeat_flag


class AudioEngine:
    """Encapsulates real-time audio capture, model inference, and metronome click playback."""

    def __init__(
        self,
        model_ckpt: str,
        sample_rate: int = 22050,
        block_size: int = 512,
        n_fft: int = 1024,
        hop_length: int = 441,
        max_context: int = 500,
        beat_freq: float = 440.0,
        downbeat_freq: float = 880.0,
        on_beat=None,
        on_downbeat=None,
    ):
        # Configuration
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.channel_count = 1
        self.on_beat = on_beat or (lambda: None)
        self.on_downbeat = on_downbeat or (lambda: None)

        # Queues for thread communication
        self.audio_q = queue.Queue()
        self.click_q = queue.Queue()
        self.stop_event = threading.Event()

        # Beat/downbeat processing pipeline
        self.processor = StreamProcessor(
            model_ckpt=model_ckpt,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            max_context=max_context,
        )

        # Pre-generate click tones
        duration = block_size / sample_rate
        self.beat_tone = make_tone(beat_freq, duration, sample_rate)
        self.downbeat_tone = make_tone(downbeat_freq, duration, sample_rate)

        # Playback scheduling index
        self.frame_index = 0

        # Build audio stream
        self.stream = sd.Stream(
            samplerate=sample_rate,
            blocksize=block_size,
            channels=self.channel_count,
            callback=self._io_callback,
            dtype='float32',
        )
        self.worker = threading.Thread(target=self._worker, daemon=True)

    def _worker(self):
        """Background worker: pulls raw audio, runs inference, schedules click events."""
        while not self.stop_event.is_set():
            try:
                chunk = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                continue
            beat, down = self.processor.process_chunk(chunk)
            if beat:
                self.on_beat()
                self.click_q.put((self.frame_index + self.block_size, self.beat_tone))
            if down:
                self.on_downbeat()
                self.click_q.put((self.frame_index + self.block_size, self.downbeat_tone))
            self.frame_index += len(chunk)

    def _io_callback(self, indata, outdata, frames, time_info, status):
        """PortAudio callback: enqueue input, mix click tones into output."""
        # Enqueue incoming audio for processing
        chunk = indata[:, 0]
        self.audio_q.put(chunk.copy())

        # Prepare output buffer
        out = np.zeros((frames,), dtype='float32')
        block_start = self.frame_index

        # Mix scheduled clicks
        while True:
            try:
                click_pos, tone = self.click_q.get_nowait()
            except queue.Empty:
                break
            rel = click_pos - block_start
            if 0 <= rel < frames:
                length = min(len(tone), frames - rel)
                out[rel : rel + length] += tone[:length]
            elif click_pos >= block_start + frames:
                self.click_q.put((click_pos, tone))
                break

        #outdata[:, 0] = out.reshape(-1, 1)
        outdata[:, 0] = out
        # No need to advance frame_index here; worker does

    def start(self):
        """Start the audio stream and worker thread."""
        self.worker.start()
        self.stream.start()

    def stop(self):
        """Stop and clean up the audio engine."""
        self.stop_event.set()
        self.worker.join()
        self.stream.stop()
        self.stream.close()


if __name__ == "__main__":
    engine = AudioEngine(model_ckpt="checkpoints/S0 shift_tolerant_weighted_bce-h128-augTrueTrueTruect=1 cc=1-v1.ckpt")
    try:
        engine.start()
        print("Running... Press Ctrl+C to stop.")
        threading.Event().wait()
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        engine.stop()