import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config, db
from .vad import StreamingVAD
from .transcriber import Transcriber
from .corrector import correct_and_condense

app = FastAPI(title="Live Speech-to-Text API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory="static"), name="static")



_transcriber: Transcriber | None = None
_executor = ThreadPoolExecutor(max_workers=4)  # runs blocking whisper/LLM calls off the event loop


@app.on_event("startup")
async def load_models():
    global _transcriber
    _transcriber = Transcriber()  


    loop = asyncio.get_event_loop()

    def _warmup():
        try:
            correct_and_condense([{"text": "hello", "low_confidence": False, "low_energy": False}])
            print("[startup] Ollama model warmed up")
        except Exception as e:
            print(f"[startup] Ollama warmup failed — is `ollama serve` running? ({e})")

    loop.run_in_executor(_executor, _warmup)


@app.post("/session")
async def new_session():
    session_id = str(uuid.uuid4())
    await db.create_session(session_id)
    return {"session_id": session_id, "ws_url": f"/ws/stream/{session_id}"}


@app.get("/transcript/{session_id}")
async def get_transcript(session_id: str):
    doc = await db.get_session(session_id)
    if not doc:
        return JSONResponse({"error": "session not found"}, status_code=404)
    return {
        "session_id": session_id,
        "status": doc["status"],
        "corrected_text": doc["corrected_text"],
        "raw_segment_count": len(doc["raw_segments"]),
    }


@app.websocket("/ws/stream/{session_id}")
async def stream_audio(websocket: WebSocket, session_id: str):

    await websocket.accept()

    existing = await db.get_session(session_id)
    if not existing:
        await db.create_session(session_id)

    loop = asyncio.get_event_loop()
    vad = StreamingVAD()
    pending_segments: list[dict] = []  # raw segments since last correction pass
    silence_timer_task: asyncio.Task | None = None

    frame_bytes_len = vad._frame_samples * config.BYTES_PER_SAMPLE
    audio_buf = bytearray()

    async def run_correction():
        nonlocal pending_segments
        if not pending_segments:
            return
        segs = pending_segments
        pending_segments = []
        corrected = await loop.run_in_executor(_executor, correct_and_condense, segs)
        if corrected:
            await db.update_corrected_text(session_id, corrected)
            try:
                await websocket.send_json({"type": "corrected", "text": corrected})
            except Exception:
                pass  # socket may already be closing; text is still saved in Mongo

    async def schedule_silence_correction():
        # if no new audio triggers a correction within CORRECT_ON_SILENCE_MS, run one anyway
        try:
            await asyncio.sleep(config.CORRECT_ON_SILENCE_MS / 1000)
            await run_correction()
        except asyncio.CancelledError:
            pass

    async def handle_segment(segment: dict):
        nonlocal pending_segments, silence_timer_task
        result = await loop.run_in_executor(_executor, _transcriber.transcribe, segment["audio"])
        cleaned = result["text"].strip()
        if not cleaned:
            return  # nothing usable (e.g. pure noise)


        normalized = cleaned.lower().strip(" .!?")
        looks_like_hallucination = (
            normalized in config.HALLUCINATION_PHRASES
            and (result["no_speech_prob"] > 0.5 or segment["duration_ms"] < config.MIN_TRUSTED_SEGMENT_MS)
        )
        if looks_like_hallucination:
            return

        low_confidence = result["low_confidence"] or segment["low_energy"]
        raw_seg = {
            "text": cleaned,
            "start_ms": None,
            "end_ms": None,
            "avg_logprob": result["avg_logprob"],
            "no_speech_prob": result["no_speech_prob"],
            "low_confidence": low_confidence,
            "low_energy": segment["low_energy"],
            "duration_ms": segment["duration_ms"],
        }


        await db.add_raw_segment(session_id, raw_seg)
        pending_segments.append(raw_seg)

        try:
            await websocket.send_json({"type": "raw_segment", **raw_seg})
        except Exception:
            pass  # socket may be closing; segment is still tracked above

        if silence_timer_task:
            silence_timer_task.cancel()
        silence_timer_task = asyncio.create_task(schedule_silence_correction())

        if len(pending_segments) >= config.CORRECT_EVERY_N_SEGMENTS:
            if silence_timer_task:
                silence_timer_task.cancel()
            await run_correction()

    try:
        while True:
            chunk = await websocket.receive_bytes()
            audio_buf.extend(chunk)

            # re-chunk arbitrary-sized incoming audio into fixed VAD frames
            while len(audio_buf) >= frame_bytes_len:
                frame = bytes(audio_buf[:frame_bytes_len])
                del audio_buf[:frame_bytes_len]
                seg = vad.push_frame(frame)
                if seg is not None:
                    await handle_segment(seg)

    except WebSocketDisconnect:
        pass
    finally:
        final_seg = vad.flush_remaining()
        if final_seg is not None:
            await handle_segment(final_seg)
        if silence_timer_task:
            silence_timer_task.cancel()
        await run_correction()
        await db.end_session(session_id)
