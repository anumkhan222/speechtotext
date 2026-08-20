import asyncio
import sys
import wave

import websockets

SERVER_WS_BASE = "ws://localhost:8000/ws/stream"
CHUNK_MS = 100  


async def stream_file(path: str, session_id: str):
    wf = wave.open(path, "rb")
    assert wf.getframerate() == 16000, "wav must be 16kHz"
    assert wf.getnchannels() == 1, "wav must be mono"
    assert wf.getsampwidth() == 2, "wav must be 16-bit PCM"

    chunk_frames = int(16000 * CHUNK_MS / 1000)

    uri = f"{SERVER_WS_BASE}/{session_id}"
    async with websockets.connect(uri) as ws:

        async def receiver():
            async for msg in ws:
                print("SERVER:", msg)

        recv_task = asyncio.create_task(receiver())

        while True:
            data = wf.readframes(chunk_frames)
            if not data:
                break
            await ws.send(data)
            await asyncio.sleep(CHUNK_MS / 1000)  

        await asyncio.sleep(3) 
        recv_task.cancel()


if __name__ == "__main__":
    import uuid
    import requests

    wav_path = sys.argv[1] if len(sys.argv) > 1 else "sample.wav"
    session = requests.post("http://localhost:8000/session").json()
    print("session:", session)
    asyncio.run(stream_file(wav_path, session["session_id"]))

    final = requests.get(f"http://localhost:8000/transcript/{session['session_id']}").json()
    print("\nFINAL TRANSCRIPT:\n", final)
