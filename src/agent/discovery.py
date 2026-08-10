"""
discovery.py -- Phase 6b: find NEW music outside the user's library.

Uses TWO Last.fm discovery surfaces, aggregated together:
1. track.getSimilar -- "tracks similar to this track"
2. artist.getSimilar + artist.getTopTracks -- "artists similar to this
   artist" → their most popular tracks

Both are seeded from library search results, aggregated across all seeds,
deduplicated, and filtered against the user's existing library so results
are genuinely new.

This is a real-time, interactive-query module (a handful of calls per
request), unlike phase3_fetch_lastfm.py's bulk checkpointed job.
"""

import os

import requests
from dotenv import load_dotenv

from search_engine import DB_FILE, find_track

LASTFM_BASE_URL = "http://ws.audioscrobbler.com/2.0/"

_session = requests.Session()


def _lastfm_call(api_key: str, method: str, **kwargs) -> dict:
    """Generic Last.fm API call wrapper."""
    params = {"method": method, "api_key": api_key, "format": "json", "autocorrect": 1, **kwargs}
    try:
        return _session.get(LASTFM_BASE_URL, params=params, timeout=10).json()
    except (requests.RequestException, ValueError):
        return {}


def get_similar_tracks(api_key: str, artist: str, track: str, limit: int = 30) -> list[dict]:
    """Fetch Last.fm's similar-tracks suggestions for one seed track."""
    resp = _lastfm_call(api_key, "track.getSimilar", artist=artist, track=track, limit=limit)
    results = []
    for t in resp.get("similartracks", {}).get("track", []):
        try:
            results.append({
                "name": t["name"],
                "artist": t["artist"]["name"],
                "match": float(t.get("match", 0)),
                "source": "similar_track",
            })
        except (KeyError, TypeError, ValueError):
            continue
    return results


def get_similar_artists(api_key: str, artist: str, limit: int = 5) -> list[str]:
    """Get artists similar to the given artist."""
    resp = _lastfm_call(api_key, "artist.getSimilar", artist=artist, limit=limit)
    artists = []
    for a in resp.get("similarartists", {}).get("artist", []):
        try:
            artists.append(a["name"])
        except (KeyError, TypeError):
            continue
    return artists


def get_artist_top_tracks(api_key: str, artist: str, limit: int = 5) -> list[dict]:
    """Get an artist's top tracks on Last.fm."""
    resp = _lastfm_call(api_key, "artist.getTopTracks", artist=artist, limit=limit)
    results = []
    for t in resp.get("toptracks", {}).get("track", []):
        try:
            results.append({
                "name": t["name"],
                "artist": t["artist"]["name"] if isinstance(t["artist"], dict) else artist,
                "match": 0.5,  # no match score from this endpoint; use a
                               # neutral default so these don't dominate over
                               # real similarity-scored results
                "source": "similar_artist_top_track",
            })
        except (KeyError, TypeError):
            continue
    return results


def _in_library(track_name: str, artist_name: str, db_path=DB_FILE) -> bool:
    """Check whether a track already exists in the user's local library."""
    candidates = find_track(track_name, artist_name, db_path=db_path)
    return len(candidates) > 0


def discover_new_music(
    seeds: list[tuple[str, str]],
    limit: int = 20,
    db_path=DB_FILE,
    exclude_artists: list[str] | None = None,
    exclude_tracks: list[str] | None = None,
) -> list[dict]:
    """Given a list of (track_name, artist_name) seed tracks, find new
    music via Last.fm, aggregated across all seeds and filtered to exclude
    anything already in the user's library.

    Uses two discovery surfaces per seed:
    1. track.getSimilar (direct track similarity)
    2. artist.getSimilar → top tracks (similar artists' best work)

    Returns a list of {'name', 'artist', 'best_match', 'seed_count', 'source'}
    sorted by seed_count (strongest signal) then best_match score, descending.
    """
    load_dotenv()
    api_key = os.getenv("LASTFM_API_KEY")
    if not api_key:
        raise EnvironmentError("LASTFM_API_KEY not found in .env")

    candidates: dict[tuple[str, str], dict] = {}

    seen_artists = set()
    for seed_track, seed_artist in seeds:
        # Surface 1: tracks similar to this specific track
        similar_tracks = get_similar_tracks(api_key, seed_artist, seed_track)
        for s in similar_tracks:
            key = (s["name"].strip().lower(), s["artist"].strip().lower())
            if key not in candidates:
                candidates[key] = {
                    "name": s["name"], "artist": s["artist"],
                    "best_match": s["match"], "seed_count": 1,
                    "source": s["source"],
                }
            else:
                candidates[key]["best_match"] = max(candidates[key]["best_match"], s["match"])
                candidates[key]["seed_count"] += 1

        # Surface 2: similar artists' top tracks (skip if we already
        # explored this artist from a prior seed to avoid redundant calls)
        artist_key = seed_artist.strip().lower()
        if artist_key not in seen_artists:
            seen_artists.add(artist_key)
            similar_artists = get_similar_artists(api_key, seed_artist)
            for sim_artist in similar_artists:
                top_tracks = get_artist_top_tracks(api_key, sim_artist)
                for t in top_tracks:
                    key = (t["name"].strip().lower(), t["artist"].strip().lower())
                    if key not in candidates:
                        candidates[key] = {
                            "name": t["name"], "artist": t["artist"],
                            "best_match": t["match"], "seed_count": 1,
                            "source": t["source"],
                        }
                    else:
                        candidates[key]["seed_count"] += 1

    # Filter out anything already in the user's library.
    new_candidates = [
        c for c in candidates.values()
        if not _in_library(c["name"], c["artist"], db_path)
    ]

    # Filter out excluded artists (case-insensitive substring match, same
    # convention as genre filtering in search_engine.py).
    if exclude_artists:
        excluded_lower = [a.strip().lower() for a in exclude_artists]
        new_candidates = [
            c for c in new_candidates
            if not any(excl in c["artist"].strip().lower() for excl in excluded_lower)
        ]

    # Filter out excluded specific tracks, same convention.
    if exclude_tracks:
        excluded_titles_lower = [t.strip().lower() for t in exclude_tracks]
        new_candidates = [
            c for c in new_candidates
            if not any(excl in c["name"].strip().lower() for excl in excluded_titles_lower)
        ]

    new_candidates.sort(key=lambda c: (c["seed_count"], c["best_match"]), reverse=True)
    return new_candidates[:limit]