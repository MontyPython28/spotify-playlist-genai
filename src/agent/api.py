"""
api.py -- L1: local web API wrapping the mood agent.

Runs a FastAPI server on 127.0.0.1 that exposes the existing agent over
HTTP, so a local browser UI (L2) can talk to it instead of the CLI. This
is a thin front door -- it imports and calls the existing, tested modules
(agent.generate_playlist, spotify_push.push_playlist) rather than
reimplementing anything.

Runs LOCALLY ONLY -- this is not a hosted service. Each person runs their
own copy on their own machine with their own keys and their own data.

Run from the project root:
    uvicorn api:app --host 127.0.0.1 --port 8000
or just:
    python api.py

Then open http://127.0.0.1:8000 in a browser (UI comes in L2; for now
the API is browsable at http://127.0.0.1:8000/docs).
"""

import os
import uuid

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from anthropic import Anthropic

from agent import generate_playlist, get_tag_vocabulary, build_search_tool
from spotify_auth import get_valid_access_token
from spotify_push import push_playlist, resolve_playlist

app = FastAPI(title="Mood Agent (local)")

# Built once at startup and reused across requests -- the vocabulary and
# search tool are static for a given database, so there's no reason to
# rebuild the ~150-tag tool schema on every query.
_state: dict = {}


def _startup():
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not found in .env")
    _state["client"] = Anthropic(api_key=api_key)
    _state["vocabulary"] = get_tag_vocabulary()
    _state["search_tool"] = build_search_tool(_state["vocabulary"])


@app.on_event("startup")
def on_startup():
    _startup()


# --- Request/response models ----------------------------------------------

class QueryRequest(BaseModel):
    prompt: str
    context: str = ""  # optional prior-turn context for refinements (L2/L3 use this)


class Track(BaseModel):
    id: str            # stable per-response ID, so the L3 editor can
                       # add/remove specific tracks unambiguously
    track_name: str
    artist_name: str
    source: str        # "library" | "new"
    genre: str = ""
    energy: str = ""
    track_uri: str | None = None  # confirmed Spotify URI (every track shown
                                    # is resolved against Spotify at query time)
    image_url: str | None = None  # album art thumbnail from Spotify
    album: str = ""


class QueryResponse(BaseModel):
    prompt: str
    tracks: list[Track]
    library_count: int
    new_count: int
    dropped_count: int = 0   # tracks the agent picked but that weren't on
                              # Spotify, so were filtered out before display
    similar_note: str | None = None
    tool_input: dict
    cost_usd: float


class PushRequest(BaseModel):
    # The tracks to push, as returned by /query (possibly edited by the
    # user in the UI -- added/removed). We take the full track list rather
    # than a query, so the user can push exactly what they see/edited.
    tracks: list[Track]
    name: str
    description: str = ""
    public: bool = False


class PushResponse(BaseModel):
    playlist_url: str
    added_count: int
    skipped: list[dict]


# --- Endpoints ------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Empty prompt")

    try:
        result = generate_playlist(
            _state["client"], _state["search_tool"], req.prompt, req.context
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

    # Resolve every track against Spotify: this both confirms the track is
    # actually ON Spotify (unresolved ones are dropped, never shown) and
    # fetches album art in the same call. Requires Spotify auth on every
    # query now -- the deliberate tradeoff for the "never show a track not
    # on Spotify" guarantee plus uniform album art.
    try:
        access_token = get_valid_access_token()
    except EnvironmentError as e:
        raise HTTPException(status_code=400, detail=f"Spotify not configured: {e}")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Spotify authorization failed: {e}")

    try:
        resolution = resolve_playlist(access_token, result["playlist"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spotify resolution failed: {e}")

    resolved_tracks = resolution["resolved"]
    dropped_count = len(resolution["dropped"])

    tracks = []
    for t in resolved_tracks:
        tracks.append(Track(
            id=uuid.uuid4().hex[:12],
            track_name=t["track_name"],
            artist_name=t["artist_name"],
            source=t["source"],
            genre=t.get("genre", "") or "",
            energy=t.get("energy", "") or "",
            track_uri=t.get("track_uri"),
            image_url=t.get("image_url"),
            album=t.get("album", "") or "",
        ))

    return QueryResponse(
        prompt=req.prompt,
        tracks=tracks,
        library_count=sum(1 for t in tracks if t.source == "library"),
        new_count=sum(1 for t in tracks if t.source == "new"),
        dropped_count=dropped_count,
        similar_note=result["similar_note"],
        tool_input=result["tool_input"],
        cost_usd=result["usage"].get("cost_usd", 0.0),
    )


@app.post("/push", response_model=PushResponse)
def push(req: PushRequest):
    if not req.tracks:
        raise HTTPException(status_code=400, detail="No tracks to push")
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Playlist name required")

    # Convert the API Track models back into the plain dict shape that
    # push_playlist expects (track_name, artist_name, track_uri, source).
    playlist = [
        {
            "track_name": t.track_name,
            "artist_name": t.artist_name,
            "track_uri": t.track_uri,
            "source": t.source,
        }
        for t in req.tracks
    ]

    try:
        access_token = get_valid_access_token()
    except EnvironmentError as e:
        raise HTTPException(status_code=400, detail=f"Spotify not configured: {e}")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Spotify authorization failed: {e}")

    try:
        result = push_playlist(access_token, playlist, req.name, req.description, req.public)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Push failed: {e}")

    return PushResponse(
        playlist_url=result["playlist_url"],
        added_count=result["added_count"],
        skipped=result["skipped"],
    )


@app.get("/health")
def health():
    return {"status": "ok", "vocabulary_size": len(_state.get("vocabulary", []))}


# --- Serve the built frontend ---------------------------------------------
# The Vite build (npm run build) emits static files into ./static. We serve
# the asset files (JS/CSS) from /assets and the index.html at the root. If
# the build hasn't been run yet, static/index.html won't exist and the root
# route returns a helpful message instead of a confusing 404.
# api.py lives at src/agent/api.py; the built frontend lands in static/ at
# the REPO ROOT (peer to src/ and frontend/), so walk up three levels:
# parent=agent, parent.parent=src, parent.parent.parent=repo root.
_STATIC_DIR = Path(__file__).parent.parent.parent / "static"

if (_STATIC_DIR / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=_STATIC_DIR / "assets"), name="assets")


@app.get("/")
def index():
    index_file = _STATIC_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(index_file)
    return {
        "message": "Frontend not built yet. Run 'npm run build' in the frontend/ "
                   "folder, or use the Vite dev server (npm run dev) during development.",
        "api_docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)