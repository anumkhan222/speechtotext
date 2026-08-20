"""
MongoDB layer (async, via motor).

Collections:
- sessions: one doc per streaming session
- segments: one doc per raw VAD-detected speech segment (audit trail / debugging)

`sessions` doc shape:
{
  _id: session_id (str),
  started_at, updated_at, ended_at,
  status: "active" | "ended",
  raw_segments: [ {text, start_ms, end_ms, confidence, low_quality, ts} ... ],
  corrected_text: "<latest cleaned + condensed transcript>",
  last_corrected_at: datetime
}
"""
from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorClient

from . import config

_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(config.MONGO_URI)
    return _client


def get_db():
    return get_client()[config.MONGO_DB_NAME]


def _now():
    return datetime.now(timezone.utc)


async def create_session(session_id: str):
    db = get_db()
    doc = {
        "_id": session_id,
        "started_at": _now(),
        "updated_at": _now(),
        "ended_at": None,
        "status": "active",
        "raw_segments": [],
        "corrected_text": "",
        "last_corrected_at": None,
    }
    await db.sessions.replace_one({"_id": session_id}, doc, upsert=True)
    return doc


async def add_raw_segment(session_id: str, segment: dict):
    db = get_db()
    segment = {**segment, "ts": _now()}
    await db.sessions.update_one(
        {"_id": session_id},
        {"$push": {"raw_segments": segment}, "$set": {"updated_at": _now()}},
    )
    await db.segments.insert_one({"session_id": session_id, **segment})
    return segment


async def update_corrected_text(session_id: str, corrected_text: str):
    db = get_db()
    await db.sessions.update_one(
        {"_id": session_id},
        {
            "$set": {
                "corrected_text": corrected_text,
                "last_corrected_at": _now(),
                "updated_at": _now(),
            }
        },
    )


async def end_session(session_id: str):
    db = get_db()
    await db.sessions.update_one(
        {"_id": session_id},
        {"$set": {"status": "ended", "ended_at": _now(), "updated_at": _now()}},
    )


async def get_session(session_id: str):
    db = get_db()
    return await db.sessions.find_one({"_id": session_id})
