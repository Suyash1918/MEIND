# MEIND

**A platform-agnostic Cognitive-Affective Gateway API.**
An externalized, on-demand "brain" for game characters, robots, and autonomous agents.

Instead of hardcoding behaviors or training heavy models locally, an entity offloads
complex problem-solving and scene understanding to MEIND. The system converts raw
environmental states into a structured, LLM-readable format, asks a large language
model what it sees, and returns a factual description the agent can act on.

---

## What works right now (v0.1)

- **Eyes** - a Three.js 3D scene rendered in the browser, sampled into a W x H grid
- **Per-pixel encoding** - each cell carries depth (Z 0-100) and color (hex),
  quantized into a 64-slot legend
- **Transport** - the grid is sent to the server as plain ASCII (RLE64 and zlib
  variants exist but hurt LLM comprehension; raw ASCII is what actually works)
- **The Screen** - the server holds a canonical world state per session. Clients
  send only deltas after the first full frame. World persists across calls.
- **LLM reading** - the world is rendered into a structured prompt with legend,
  grid, recent deltas, and a strict 4-section output format
- **Streaming responses** - the LLM writes back token-by-token into a "thought bubble"
- **Manual snapshots** - a button fires a snapshot whenever you want the LLM to look

---

## Architecture

    CLIENT                     MEIND SERVER                    LLM
      |                            |                            |
      |  full frame  (once)        |                            |
      |--------------------------->|                            |
      |                            |  stores world              |
      |                            |  (the "screen")            |
      |                            |                            |
      |  delta (on snapshot click) |                            |
      |--------------------------->|                            |
      |                            |  paints delta onto screen  |
      |                            |  updates rolling log       |
      |                            |                            |
      |  "describe what you see"   |                            |
      |--------------------------->|  renders screen to text    |
      |                            |--------------------------->|
      |                            |                            |
      |                            |  <---- streamed answer     |
      |  <---- streamed answer     |                            |

The **screen** is the canonical state. MEIND owns it. The LLM never sees deltas
directly - it sees the current full world plus a short trail of recent motion.

---

## Files

| File | Purpose |
|---|---|
| `server.py` | FastAPI app. Endpoints for session lifecycle and LLM streaming. |
| `session.py` | The "screen": holds world state, deltas, and builds the LLM prompt. |
| `static/index.html` | Browser client: 3D scene, grid sampler, delta stream, thought bubble. |
| `requirements.txt` | Python dependencies. |
| `.env` | Secrets. **Never committed.** |

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET  | `/` | The browser client |
| POST | `/vision` | Legacy raw-frame endpoint |
| POST | `/session/start` | Begin a session, send first full frame |
| POST | `/session/delta` | Send changed cells since last frame |
| POST | `/session/ask_stream` | Ask the LLM to describe the current world (streamed) |
| GET  | `/session/prompt_preview` | Debug: shows the exact prompt sent to the LLM |
| GET  | `/session/info` | Session metadata |

---

## Setup

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt

Create a `.env` file:

    GROQ_API_KEY=gsk_your_key_here
    MEIND_MODEL=openai/gpt-oss-120b

Run:

    python server.py

Open http://127.0.0.1:8000 in a browser.

---

## How to use

1. Click **Start** - creates a session and sends the first full frame
2. The scene animates on the left
3. Click **Send Snapshot** whenever you want MEIND to look
4. The thought bubble streams the LLM's description of the world in real time

To see what the LLM actually receives:

    http://127.0.0.1:8000/session/prompt_preview?session_id=YOUR_SID

---

## Known constraints

- **No card, no money.** Runs on Groq's free tier.
- **Free-tier rate limits.** Groq caps around 8K tokens/min on free. One snapshot
  per ~60 seconds. The snapshot is manual for exactly this reason.
- **Model choice matters.** `openai/gpt-oss-120b` works. `gpt-oss-20b` cannot
  reliably decompress the grid and hallucinates. DeepSeek via NVIDIA NIM was
  attempted and is currently down.
- **Plain ASCII over compressed formats.** RLE64 and zlib save tokens but cost
  comprehension. Raw ASCII is what the model actually reads.

---

## Roadmap

- [ ] Code generation - the LLM outputs executable snippets, not just descriptions
- [ ] Self-correction - failed actions are fed back as new deltas
- [ ] Reflex storage - successful solutions cached client-side
- [ ] Quiella Engine - affective filtering (valence + intensity) before reasoning
- [ ] Object-hint layer - shape names added to the legend

---

## License

Personal project. Not yet released.