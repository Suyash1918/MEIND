from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn, time, sys

BASE = Path(__file__).parent
load_dotenv(BASE / ".env")

import session as meind_session

STATIC = BASE / "static"
LAST_FRAME = BASE / "last_frame.txt"

def log(*args):
    print("[SRV]", *args, file=sys.stderr, flush=True)

app = FastAPI(title="MEIND Eyes + Session")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.post("/vision")
async def vision(request: Request):
    payload = await request.json()
    text = payload.get("data", "")
    LAST_FRAME.write_text(text, encoding="utf-8")
    return {"ok": True, "encoding": payload.get("encoding"), "wire": len(text)}


@app.post("/session/start")
async def session_start(request: Request):
    p = await request.json()
    w, h = int(p["width"]), int(p["height"])
    rows = p["grid"]
    legend = p.get("legend", "")
    log(f"/session/start  {w}x{h}  rows={len(rows)}  legend={len(legend)}ch")
    if len(rows) != h or any(len(r) != w for r in rows):
        log("  shape mismatch!")
        return JSONResponse({"error": "grid shape mismatch"}, status_code=400)
    s = meind_session.new_session(w, h, "\n".join(rows), legend)
    return {"session_id": s.id, "width": w, "height": h}


@app.post("/session/delta")
async def session_delta(request: Request):
    p = await request.json()
    sid = p.get("session_id", "")
    s = meind_session.get_session(sid)
    if s is None:
        log(f"/session/delta  UNKNOWN session {sid[:8]}")
        return JSONResponse({"error": "no such session"}, status_code=404)
    s.apply_delta(p.get("changes", []), note=p.get("note", ""))
    log(f"/session/delta  sid={sid[:8]}  changes={len(p.get('changes', []))}  recent={len(s.recent)}")
    return {"ok": True, "recent": len(s.recent)}


@app.post("/session/ask_stream")
async def session_ask_stream(request: Request):
    p = await request.json()
    sid = p.get("session_id", "")
    s = meind_session.get_session(sid)
    if s is None:
        log(f"/session/ask_stream  UNKNOWN session {sid[:8]}")
        return JSONResponse({"error": "no such session"}, status_code=404)

    q = p.get("question", "Describe everything you see in front of you right now.")
    s.turn_count += 1
    log(f"/session/ask_stream  sid={sid[:8]}  turn={s.turn_count}  q={len(q)}ch")

    t0 = time.time()
    prompt = s.render_prompt(q)
    log(f"  prompt built: {len(prompt)} chars in {time.time()-t0:.2f}s")

    def generate():
        total = 0
        try:
            for chunk in meind_session.ask_llm_stream(prompt):
                total += len(chunk)
                yield chunk
        except Exception as e:
            log(f"  generator exception: {type(e).__name__}: {e}")
            yield f"\n[error] {type(e).__name__}: {e}"
        log(f"  /session/ask_stream DONE  {total} chars in {time.time()-t0:.2f}s")

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


@app.get("/session/info")
async def session_info(session_id: str):
    s = meind_session.get_session(session_id)
    if s is None:
        return JSONResponse({"error": "no such session"}, status_code=404)
    return {
        "id": s.id, "size": [s.width, s.height],
        "recent_count": len(s.recent), "turn_count": s.turn_count,
        "last_delta_t": s.last_delta_t,
    }


@app.get("/session/prompt_preview")
async def session_prompt_preview(session_id: str, question: str = "Describe what you see."):
    s = meind_session.get_session(session_id)
    if s is None:
        return JSONResponse({"error": "no such session"}, status_code=404)
    return {"prompt": s.render_prompt(question)}


if __name__ == "__main__":
    log("MEIND server starting...")
    log(f"  GROQ_API_KEY: {'set (' + str(len(meind_session.GROQ_API_KEY)) + ' chars)' if meind_session.GROQ_API_KEY else 'MISSING'}")
    log(f"  model: {meind_session.GROQ_MODEL}")
    log(f"  max_tokens: {meind_session.GROQ_MAX_TOKENS}")
    log(f"  reasoning: {meind_session.GROQ_REASONING or '(default)'}")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")