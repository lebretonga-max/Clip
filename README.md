# Clipper backend

## Prérequis
- Python 3.10+
- ffmpeg (avec libass) : `sudo apt install ffmpeg` / `brew install ffmpeg`

## Lancer
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```
Variables : `WHISPER_MODEL` (tiny|base|small|medium|large-v3, défaut small), `DATA_DIR` (défaut ./data).

## API
- `POST /jobs` (multipart : `file`, `n_clips`=5, `min_s`=20, `max_s`=55) → `{id}`
- `GET /jobs/{id}` → `status` (queued|transcribing|done|error), `language`, `segments` (mots horodatés), `clips` (start, end, score, title)
- `POST /jobs/{id}/export?start=12.4&end=47&x=0.5&subtitles=true` → MP4 1080x1920 sous-titré

## Test rapide
```bash
curl -F file=@video.mp4 localhost:8000/jobs
curl localhost:8000/jobs/<id>
curl -X POST "localhost:8000/jobs/<id>/export?start=10&end=45" -o clip.mp4
```

## Limites
- Détection des moments forts = heuristique (débit de parole + ponctuation). À remplacer par un scoring LLM plus tard.
- Recadrage fixe (x), pas de suivi du visage.
- Jobs traités en tâche de fond dans le même process : pour la prod, ajouter une file (Celery/RQ) et un stockage objet.
