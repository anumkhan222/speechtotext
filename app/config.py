SAMPLE_RATE = 16000
FRAME_MS = 32
BYTES_PER_SAMPLE = 2

VAD_THRESHOLD = 0.5
VAD_MIN_SILENCE_MS = 800
VAD_MIN_SPEECH_MS = 250
LOW_ENERGY_RMS_THRESHOLD = 300

WHISPER_MODEL_SIZE = "small"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"
WHISPER_LOW_CONF_LOGPROB = -1.0
WHISPER_HIGH_NO_SPEECH_PROB = 0.6

HALLUCINATION_PHRASES = {
    "thank you",
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "subscribe",
}

MIN_TRUSTED_SEGMENT_MS = 600

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"

CORRECT_EVERY_N_SEGMENTS = 4
CORRECT_ON_SILENCE_MS = 2500

MONGO_URI = "mongodb+srv://anumkh256_db_user:sc07TzF0ueLiLqr8@cluster0.whlcuvr.mongodb.net/"
MONGO_DB_NAME = "speechtext"