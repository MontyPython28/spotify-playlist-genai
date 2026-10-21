"""
extra_sources.py -- live new-music discovery from Deezer and ListenBrainz,
used by discovery.py alongside Last.fm. All free, no API keys needed.

Every function returns candidates in the same shape discovery.py's
_merge() expects: {'name', 'artist', 'match' (0-1), 'source'}.
Every function fails soft (returns []) so one flaky source never breaks
a playlist request.
"""

import os
import threading
import time

import requests

DEEZER_BASE = "https://api.deezer.com"
MB_BASE = "https://musicbrainz.org/ws/2"
LB_LABS = "https://labs.api.listenbrainz.org"

# ListenBrainz's published similarity datasets. If they rename one, the
# call just returns [] -- check https://labs.api.listenbrainz.org for current names.
LB_API = "https://api.listenbrainz.org/1"
LB_SIMILAR_ARTISTS_ALGO = (
    "session_based_days_7500_session_300_contribution_5_threshold_10_limit_100_filter_True_skip_30"
)

_session = requests.Session()
_session.headers["User-Agent"] = "SpotifyMoodAgent/1.0 ( " + os.getenv(
    "MUSICBRAINZ_CONTACT", "https://github.com/MontyPython28/spotify-playlist-genai"
) + " )"

_mb_lock = threading.Lock()
_mb_last = [0.0]


def _get(url: str, **kwargs):
    try:
        resp = _session.get(url, timeout=10, **kwargs)
        return resp.json() if resp.ok else {}
    except (requests.RequestException, ValueError):
        return {}


# ---------------------------------------------------------------- Deezer
def _deezer_artist_id(artist: str) -> int | None:
    hits = _get(f"{DEEZER_BASE}/search/artist", params={"q": artist, "limit": 1}).get("data", [])
    return hits[0]["id"] if hits else None


def deezer_related_top_tracks(artist: str, n_artists: int = 4, per_artist: int = 4) -> list[dict]:
    """Seed artist -> Deezer related artists -> each one's top tracks."""
    aid = _deezer_artist_id(artist)
    if not aid:
        return []
    related = _get(f"{DEEZER_BASE}/artist/{aid}/related",
                   params={"limit": n_artists}).get("data", [])[:n_artists]
    results = []
    for rank, rel in enumerate(related):
        top = _get(f"{DEEZER_BASE}/artist/{rel['id']}/top",
                   params={"limit": per_artist}).get("data", [])
        for t in top:
            try:
                results.append({
                    "name": t["title"],
                    "artist": t["artist"]["name"],
                    "match": round(0.8 - rank * 0.1, 2),  # closer artists score higher
                    "source": "deezer_related",
                })
            except (KeyError, TypeError):
                continue
    return results


# ------------------------------------------------------------ ListenBrainz
def _mb_artist_id(artist: str) -> str | None:
    """MusicBrainz artist lookup, serialised to respect its 1 req/sec rule."""
    with _mb_lock:
        wait = _mb_last[0] + 1.05 - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _mb_last[0] = time.monotonic()
        res = _get(f"{MB_BASE}/artist/", params={"query": f'artist:"{artist}"', "fmt": "json", "limit": 1})
    arts = res.get("artists", []) if res else []
    return arts[0]["id"] if arts and float(arts[0].get("score", 0)) >= 90 else None


def listenbrainz_similar_artist_tracks(artist: str, n_artists: int = 4, per_artist: int = 4) -> list[dict]:
    """Seed artist -> ListenBrainz similar artists (from real listening
    sessions) -> each one's most-played recordings."""
    amb = _mb_artist_id(artist)
    if not amb:
        return []
    rows = _get(f"{LB_LABS}/similar-artists/json",
                params={"artist_mbids": amb, "algorithm": LB_SIMILAR_ARTISTS_ALGO})
    if not isinstance(rows, list):
        return []
    rows = [r for r in rows if isinstance(r, dict) and r.get("artist_mbid")][:n_artists]
    results = []
    for rank, r in enumerate(rows):
        top = _get(f"{LB_API}/popularity/top-recordings-for-artist/{r['artist_mbid']}")
        if not isinstance(top, list):
            continue
        for t in top[:per_artist]:
            if t.get("recording_name"):
                results.append({
                    "name": t["recording_name"],
                    "artist": t.get("artist_name") or r.get("name", ""),
                    "match": round(0.8 - rank * 0.1, 2),
                    "source": "listenbrainz_similar",
                })
    return results
