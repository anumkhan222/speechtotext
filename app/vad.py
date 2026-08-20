import numpy as np
import torch

from . import config


class StreamingVAD:
    def __init__(self):
        
        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self.model.eval()

        self._speech_buf: list[np.ndarray] = []
        self._silence_ms = 0
        self._speech_ms = 0
        self._in_speech = False

        self._frame_samples = int(config.SAMPLE_RATE * config.FRAME_MS / 1000)

    def _frame_prob(self, frame_i16: np.ndarray) -> float:
        audio = torch.from_numpy(frame_i16.astype(np.float32) / 32768.0)
        with torch.no_grad():
            return float(self.model(audio, config.SAMPLE_RATE).item())

    def push_frame(self, frame_bytes: bytes):

        frame_i16 = np.frombuffer(frame_bytes, dtype=np.int16)
        if len(frame_i16) == 0:
            return None

        prob = self._frame_prob(frame_i16)
        is_speech = prob >= config.VAD_THRESHOLD

        if is_speech:
            self._speech_buf.append(frame_i16)
            self._speech_ms += config.FRAME_MS
            self._silence_ms = 0
            self._in_speech = True
            return None

        if self._in_speech:
            self._silence_ms += config.FRAME_MS
            self._speech_buf.append(frame_i16)
            if self._silence_ms >= config.VAD_MIN_SILENCE_MS:
                return self._flush()
        return None

    def _flush(self):
        if not self._speech_buf or self._speech_ms < config.VAD_MIN_SPEECH_MS:
            self._reset()
            return None

        trailing_silence_frames = config.VAD_MIN_SILENCE_MS // config.FRAME_MS
        frames = self._speech_buf[: max(1, len(self._speech_buf) - trailing_silence_frames)]
        audio_i16 = np.concatenate(frames) if frames else np.concatenate(self._speech_buf)

        rms = float(np.sqrt(np.mean(audio_i16.astype(np.float64) ** 2)))
        duration_ms = int(len(audio_i16) / config.SAMPLE_RATE * 1000)

        segment = {
            "audio": audio_i16.astype(np.float32) / 32768.0,
            "rms": rms,
            "low_energy": rms < config.LOW_ENERGY_RMS_THRESHOLD,
            "duration_ms": duration_ms,
        }
        self._reset()
        return segment

    def flush_remaining(self):
        if self._in_speech:
            return self._flush()
        return None

    def _reset(self):
        self._speech_buf = []
        self._silence_ms = 0
        self._speech_ms = 0
        self._in_speech = False