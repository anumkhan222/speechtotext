import json
import re
import requests

from . import config

SYSTEM_PROMPT = """You are a grammar correction assistant.

You will receive raw speech-to-text text.

Rules:

1. Correct grammar, spelling, punctuation, capitalization, and sentence structure.
2. Preserve the original meaning exactly.
3. Do NOT add, remove, summarize, or rewrite information.
4. Do NOT invent missing words, facts, names, numbers, places, or details.
5. Remove obvious speech disfluencies only when they are clearly accidental (e.g., "I I", "the the", "um", "uh") without changing the meaning.
6. If a word or phrase is unclear, leave it as close to the original as possible instead of guessing.
7. Keep all information that appears in the input.

Output format:

- Output ONLY the corrected text.
- Do not include explanations, labels, quotation marks, or markdown.
- The first character of your response must be the first character of the corrected text.
"""

_PREAMBLE_RE = re.compile(
    r"^(here'?s|here is|sure,?|okay,?|cleaned([\s\-]up)?( text| version)?:?)\b[^:\n]*:\s*",
    re.IGNORECASE,
)


def _strip_preamble(text: str) -> str:
   
    text = text.strip()
    text = _PREAMBLE_RE.sub("", text).strip()
    if len(text) >= 2 and text[0] in "\"'" and text[-1] in "\"'":
        text = text[1:-1].strip()
    return text


def _build_user_prompt(segments: list[dict]) -> str:
    lines = []
    for s in segments:
        flags = []
        if s.get("low_confidence"):
            flags.append("low_confidence")
        if s.get("low_energy"):
            flags.append("low_energy")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        lines.append(f"- {s['text']}{flag_str}")
    return "Raw segments:\n" + "\n".join(lines)


def correct_and_condense(segments: list[dict]) -> str:

    if not segments:
        return ""

    prompt = SYSTEM_PROMPT + "\n\n" + _build_user_prompt(segments)

    resp = requests.post(
        config.OLLAMA_URL,
        json={
            "model": config.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return _strip_preamble(data.get("response", ""))