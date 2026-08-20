"""
faster-whisper (Whisper via CTranslate2) wrapper.
Model loads once, at process start — free, fully local, no API calls.
"""
import numpy as np
from faster_whisper import WhisperModel

from . import config


class Transcriber:
    def __init__(self):
        self.model = WhisperModel(
            config.WHISPER_MODEL_SIZE,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )

    def transcribe(self, audio_f32: np.ndarray) -> dict:
        """
        audio_f32: mono float32 PCM at config.SAMPLE_RATE, range [-1, 1]
        Returns {"text": str, "avg_logprob": float, "no_speech_prob": float, "low_confidence": bool}
        """
        segments, info = self.model.transcribe(
            audio_f32,
            language="en",          # drop this arg for auto language detection
            vad_filter=False,        # we already did VAD ourselves upstream
            beam_size=5,
        )
        segments = list(segments)
        if not segments:
            return {"text": "", "avg_logprob": -999.0, "no_speech_prob": 1.0, "low_confidence": True}

        text = " ".join(s.text.strip() for s in segments).strip()
        avg_logprob = sum(s.avg_logprob for s in segments) / len(segments)
        no_speech_prob = max(s.no_speech_prob for s in segments)

        low_confidence = (
            avg_logprob < config.WHISPER_LOW_CONF_LOGPROB
            or no_speech_prob > config.WHISPER_HIGH_NO_SPEECH_PROB
        )

        return {
            "text": text,
            "avg_logprob": avg_logprob,
            "no_speech_prob": no_speech_prob,
            "low_confidence": low_confidence,
        }
