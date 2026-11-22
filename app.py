"""Clipper backend : upload -> transcription (Whisper) -> clips suggérés -> export MP4 9:16 sous-titré."""
import json, os, re, shutil, subprocess, uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

DATA = Path(os.getenv("DATA_DIR", "data"))
DATA.mkdir(exist_ok=True)
MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")  # tiny / base / small / medium / large-v3

app = FastAPI(title="Clipper")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_model = None


def model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(MODEL_SIZE, compute_type="int8")  # CPU ; "float16" si GPU
    return _model


# ---------- état des jobs (fichier JSON par job) ----------
def jpath(job_id): return DATA / job_id / "job.json"


def load(job_id):
    p = jpath(job_id)
    if not p.exists():
        raise HTTPException(404, "Job introuvable")
    return json.loads(p.read_text())


def save(job_id, **kw):
    p = jpath(job_id)
    job = json.loads(p.read_text()) if p.exists() else {"id": job_id}
    job.update(kw)
    p.write_text(json.dumps(job, ensure_ascii=False))


# ---------- pipeline ----------
def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-800:])


def transcribe(video: Path):
    wav = video.parent / "audio.wav"
    run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", str(wav)])
    segs, info = model().transcribe(str(wav), word_timestamps=True, vad_filter=True)
    out = []
    for s in segs:
        out.append({
            "start": s.start, "end": s.end, "text": s.text.strip(),
            "words": [{"start": w.start, "end": w.end, "word": w.word.strip()} for w in (s.words or [])],
        })
    return out, info.language


def suggest_clips(segments, n=5, min_s=20, max_s=55):
    """Fenêtres alignées sur les phrases, notées par débit de parole + ponctuation expressive."""
    cands = []
    for i, a in enumerate(segments):
        for j in range(i, len(segments)):
            b = segments[j]
            dur = b["end"] - a["start"]
            if dur > max_s:
                break
            if dur < min_s:
                continue
            txt = " ".join(s["text"] for s in segments[i:j + 1])
            words = len(txt.split())
            score = words / dur + 0.15 * (txt.count("?") + txt.count("!"))
            cands.append({"start": a["start"], "end": b["end"], "score": round(score, 3),
                          "title": (segments[i]["text"][:60]).strip()})
    cands.sort(key=lambda c: -c["score"])
    picked = []
    for c in cands:  # pas de chevauchement
        if all(c["end"] <= p["start"] or c["start"] >= p["end"] for p in picked):
            picked.append(c)
        if len(picked) == n:
            break
    return sorted(picked, key=lambda c: c["start"])


def process(job_id, n, min_s, max_s):
    try:
        video = next((DATA / job_id).glob("source.*"))
        save(job_id, status="transcribing")
        segs, lang = transcribe(video)
        save(job_id, status="done", language=lang, segments=segs,
             clips=suggest_clips(segs, n, min_s, max_s))
    except Exception as e:
        save(job_id, status="error", error=str(e))


def srt_for(segments, start, end, path: Path, words_per_cue=4):
    def ts(t):
        t = max(0, t)
        return f"{int(t//3600):02}:{int(t%3600//60):02}:{int(t%60):02},{int(t%1*1000):03}"
    words = [w for s in segments for w in s["words"] if w["end"] > start and w["start"] < end]
    cues = []
    for k in range(0, len(words), words_per_cue):
        g = words[k:k + words_per_cue]
        cues.append(f"{len(cues)+1}\n{ts(g[0]['start']-start)} --> {ts(min(g[-1]['end'], end)-start)}\n"
                    + " ".join(w["word"] for w in g) + "\n")
    path.write_text("\n".join(cues), encoding="utf-8")


# ---------- API ----------
@app.post("/jobs")
async def create_job(bg: BackgroundTasks, file: UploadFile = File(...),
                     n_clips: int = Form(5), min_s: int = Form(20), max_s: int = Form(55)):
    job_id = uuid.uuid4().hex[:10]
    d = DATA / job_id
    d.mkdir()
    with open(d / f"source{Path(file.filename).suffix or '.mp4'}", "wb") as f:
        shutil.copyfileobj(file.file, f)
    save(job_id, status="queued")
    bg.add_task(process, job_id, n_clips, min_s, max_s)
    return {"id": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    return load(job_id)


@app.post("/jobs/{job_id}/export")
def export(job_id: str, start: float, end: float, x: float = 0.5, subtitles: bool = True):
    """Découpe [start, end], recadre en 9:16 (x = 0..1 position horizontale), brûle les sous-titres."""
    job = load(job_id)
    if job.get("status") != "done" or end - start < 0.5:
        raise HTTPException(400, "Job non terminé ou intervalle invalide")
    d = DATA / job_id
    src = next(d.glob("source.*"))
    out = d / f"clip_{int(start*1000)}_{int(end*1000)}.mp4"
    vf = f"crop=ih*9/16:ih:(iw-ih*9/16)*{x}:0,scale=1080:1920"
    if subtitles:
        srt = d / "clip.srt"
        srt_for(job["segments"], start, end, srt)
        style = "FontName=Arial,FontSize=9,Bold=1,Outline=2,Alignment=2,MarginV=45"
        vf += f",subtitles=clip.srt:force_style='{style}'"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(start), "-i", src.name, "-t", str(end - start),
             "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
             "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out.name],
            cwd=d, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(500, e.stderr.decode()[-800:])
    return FileResponse(out, media_type="video/mp4", filename=out.name)
