"""
MEIND Session - Groq backend with streaming.
Plain ASCII grid (no compression). Manual snapshots.
"""
from __future__ import annotations
import os, time, uuid, sys
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Iterator

try:
    from groq import Groq
except ImportError:
    Groq = None


def log(*args):
    print("[MEIND]", *args, file=sys.stderr, flush=True)


# -------- Config --------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.environ.get("MEIND_MODEL", "openai/gpt-oss-120b")
GROQ_MAX_TOKENS = int(os.environ.get("MEIND_MAX_TOKENS", "8192"))
GROQ_TEMPERATURE = float(os.environ.get("MEIND_TEMPERATURE", "0.5"))
GROQ_REASONING = os.environ.get("MEIND_REASONING", "").strip().lower()

MAX_RECENT_DELTAS = 50

SYSTEM_PROMPT = (
    "You are a game character describing what you see through your own eyes. "
    "Be factual, specific, and detailed. Never invent objects that are not in the grid. "
    "Follow the requested section format exactly. Write the complete report."
)


# -------- Delta --------
@dataclass
class Delta:
    t: float
    changes: list
    note: str = ""
    width: int = 192

    def render(self) -> str:
        if not self.changes:
            return f"[t={self.t:.2f}] no change"
        w = self.width
        xs = [i % w for i, _ in self.changes]
        ys = [i // w for i, _ in self.changes]
        counts = {}
        for _, c in self.changes:
            counts[c] = counts.get(c, 0) + 1
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:6]
        top_s = " ".join(f"{c}x{n}" for c, n in top)
        return (f"[t={self.t:.2f}] {len(self.changes)} cells changed "
                f"in x=[{min(xs)}..{max(xs)}] y=[{min(ys)}..{max(ys)}]. "
                f"chars: {top_s}" + (f" | {self.note}" if self.note else ""))


# -------- Session --------
@dataclass
class Session:
    id: str
    width: int
    height: int
    created: float = field(default_factory=time.time)
    world: list = field(default_factory=list)
    legend: str = ""
    recent: deque = field(default_factory=lambda: deque(maxlen=MAX_RECENT_DELTAS))
    last_delta_t: float = 0.0
    turn_count: int = 0

    def set_full_frame(self, width, height, grid_text, legend):
        self.width = width
        self.height = height
        self.legend = legend
        flat = []
        for line in grid_text.splitlines():
            flat.extend(list(line))
        self.world = flat
        self.turn_count = 0
        self.recent.clear()

    def apply_delta(self, changes, note=""):
        now = time.time() - self.created
        normalized = []
        if not isinstance(changes, list):
            changes = []
        if changes and not isinstance(changes[0], (list, tuple)):
            for k in range(0, len(changes) - 1, 2):
                try:
                    normalized.append((int(changes[k]), str(changes[k + 1])))
                except (ValueError, TypeError):
                    pass
        else:
            for item in changes:
                try:
                    i, c = item
                    normalized.append((int(i), str(c)))
                except (ValueError, TypeError):
                    pass
        for idx, ch in normalized:
            if 0 <= idx < len(self.world):
                self.world[idx] = ch
        self.recent.append(Delta(t=now, changes=normalized, note=note, width=self.width))
        self.last_delta_t = now

    def render_world_raw(self):
        lines = []
        for y in range(self.height):
            start = y * self.width
            lines.append("".join(self.world[start:start + self.width]))
        return "\n".join(lines)

    def _recent_text(self):
        if not self.recent:
            return "(no changes yet)"
        return "\n".join(d.render() for d in self.recent)

    def render_prompt(self, question):
        world_block = self.render_world_raw()
        legend = self.legend or "(no legend provided)"
        return f"""You are a game character - an agent in a 3D world. You see through your own eyes.
You are looking forward. This is what is in front of you right now.

HOW TO READ THE DATA
--------------------
You receive a grid of characters. Each cell is one pixel of vision.
The grid is {self.width} columns wide and {self.height} rows tall.
Reading order: left to right, top to bottom.

Each character maps to a (depth, color) pair via the LEGEND below.
Depth Z: 0 = touching you. 100 = farthest you can see.

LEGEND (character -> Z-range, color, pixel count)
-------------------------------------------------
{legend}

CURRENT WORLD (plain ASCII grid, {self.width} cols x {self.height} rows)
-------------------------------------------------
{world_block}

RECENT CHANGES (oldest to newest)
---------------------------------
{self._recent_text()}

QUESTION
--------
{question}

Reply as the character, in first person ("I see..."). Fill in ALL four sections below.
Write the complete report. Do not stop early.

## Overall Layout
2-3 sentences. Where is the sky, the ground, the horizon? What dominates the frame?

## Objects I See
One bullet per major object. For each: what it is (shape + color), where it is
(left / center / right and near / mid / far), and its approximate size in the frame.
Aim for 5-10 bullets.

## Depth Order (nearest to farthest)
A ranked list of what is closest to me, what sits in the middle distance, and
what is farthest away.

## What Has Changed
1-3 sentences describing any motion or change visible from recent deltas.
If nothing changed, say so.

Do not narrate the format. Do not repeat the legend back to me."""


SESSIONS = {}

def new_session(width, height, grid_text, legend):
    sid = str(uuid.uuid4())
    s = Session(id=sid, width=width, height=height)
    s.set_full_frame(width, height, grid_text, legend)
    SESSIONS[sid] = s
    log(f"new session {sid[:8]}  {width}x{height}  legend {len(legend)} chars")
    return s

def get_session(sid):
    return SESSIONS.get(sid)


# -------- Groq --------
_client = None

def _get_client():
    global _client
    if Groq is None:
        raise RuntimeError("groq package not installed. Run: pip install groq")
    if _client is None:
        log(f"init Groq client  key={'set' if GROQ_API_KEY else 'MISSING'}")
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY not set. Put it in .env")
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def ask_llm_stream(prompt) -> Iterator[str]:
    log(f"ask_llm_stream START  prompt={len(prompt)} chars  model={GROQ_MODEL}")

    try:
        client = _get_client()
    except Exception as e:
        log(f"client init failed: {e}")
        yield f"[config error] {e}"
        return

    kwargs = dict(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=GROQ_TEMPERATURE,
        max_tokens=GROQ_MAX_TOKENS,
        stream=True,
    )
    if GROQ_REASONING:
        kwargs["reasoning_effort"] = GROQ_REASONING

    log(f"calling completions.create  max_tokens={GROQ_MAX_TOKENS}  reasoning={GROQ_REASONING or 'default'}")

    t0 = time.time()
    try:
        stream = client.chat.completions.create(**kwargs)
        log(f"stream object received after {time.time()-t0:.2f}s")
    except Exception as e:
        log(f"create() FAILED after {time.time()-t0:.2f}s: {type(e).__name__}: {e}")
        yield f"\n[stream error] {type(e).__name__}: {e}"
        return

    chunk_count = 0
    char_count = 0
    try:
        for chunk in stream:
            chunk_count += 1
            if chunk_count == 1:
                log(f"first chunk after {time.time()-t0:.2f}s")
            try:
                delta = chunk.choices[0].delta
                txt = getattr(delta, "content", None)
            except Exception:
                txt = None
            if txt:
                char_count += len(txt)
                yield txt
        log(f"stream DONE  chunks={chunk_count}  chars={char_count}  total={time.time()-t0:.2f}s")
    except Exception as e:
        log(f"stream iteration FAILED after {time.time()-t0:.2f}s: {type(e).__name__}: {e}")
        yield f"\n[stream error] {type(e).__name__}: {e}"