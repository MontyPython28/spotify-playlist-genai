"""
spotify_push.py -- push a blended playlist (from playlist.py) to Spotify.

Library tracks already have a real track_uri (from your original Spotify
export). Discovery ("new") tracks only have a name + artist from Last.fm,
so each one needs a Search API lookup to resolve a real Spotify URI
before it can be added -- some may not be found (imperfect title/artist
match, not on Spotify, etc.) and are reported as skipped rather than
silently dropped.
"""

import requests
from concurrent.futures import ThreadPoolExecutor

API_BASE = "https://api.spotify.com/v1"


def _auth_headers(access_token: str) -> dict:
    return {"Authorization": f"Bearer {access_token}"}


def _raise_with_detail(resp: requests.Response) -> None:
    """requests' default raise_for_status() only surfaces the status code,
    discarding the response body -- but Spotify's error responses almost
    always include a specific 'message' explaining WHY (insufficient
    scope, app not authorized for this user, etc.), which is exactly the
    information needed to diagnose failures like a 403 on add-tracks.
    This re-raises with that detail included instead of just the code."""
    if resp.status_code >= 400:
        try:
            detail = resp.json()
        except ValueError:
            detail = resp.text
        raise requests.exceptions.HTTPError(
            f"{resp.status_code} error from {resp.url}\nResponse body: {detail}",
            response=resp,
        )


def get_current_user_id(access_token: str) -> str:
    resp = requests.get(f"{API_BASE}/me", headers=_auth_headers(access_token))
    _raise_with_detail(resp)
    return resp.json()["id"]


def create_playlist(access_token: str, name: str, description: str = "", public: bool = False) -> dict:
    """Create a playlist for the current user. Returns the playlist object
    (includes 'id', 'external_urls.spotify' for the shareable link)."""
    resp = requests.post(
        f"{API_BASE}/me/playlists",
        headers={**_auth_headers(access_token), "Content-Type": "application/json"},
        json={"name": name, "description": description, "public": public},
    )
    _raise_with_detail(resp)
    return resp.json()


def search_track(access_token: str, track_name: str, artist_name: str) -> str | None:
    """Look up a track by name + artist, return its Spotify URI, or None
    if no confident match was found."""
    query = f"track:{track_name} artist:{artist_name}"
    resp = requests.get(
        f"{API_BASE}/search",
        headers=_auth_headers(access_token),
        params={"q": query, "type": "track", "limit": 1},
    )
    _raise_with_detail(resp)
    items = resp.json().get("tracks", {}).get("items", [])
    if not items:
        return None
    return items[0]["uri"]


def resolve_track(access_token: str, track_name: str, artist_name: str) -> dict | None:
    """Like search_track, but returns the fuller match info the UI needs --
    URI plus album artwork and the canonical name/artist Spotify has on
    file. Returns None if no confident match (i.e. the track isn't on
    Spotify), which is exactly the signal used to filter it out of results
    before the user ever sees it.

    Returns:
        {
          "uri": str,
          "image_url": str | None,   # smallest-but-decent album thumbnail
          "album": str,
          "spotify_name": str,       # Spotify's canonical track name
          "spotify_artist": str,     # Spotify's canonical primary artist
        }
    """
    query = f"track:{track_name} artist:{artist_name}"
    resp = requests.get(
        f"{API_BASE}/search",
        headers=_auth_headers(access_token),
        params={"q": query, "type": "track", "limit": 1},
    )
    _raise_with_detail(resp)
    items = resp.json().get("tracks", {}).get("items", [])
    if not items:
        return None

    item = items[0]
    album = item.get("album", {})
    images = album.get("images", [])
    # Spotify returns images largest-first (typically 640/300/64). Prefer a
    # mid/small one for a list thumbnail rather than the full-size cover.
    image_url = images[-1]["url"] if images else None
    if len(images) >= 2:
        image_url = images[1]["url"]  # the ~300px middle size when available

    artists = item.get("artists", [])
    return {
        "uri": item["uri"],
        "image_url": image_url,
        "album": album.get("name", ""),
        "spotify_name": item.get("name", track_name),
        "spotify_artist": artists[0]["name"] if artists else artist_name,
    }


def add_tracks_to_playlist(access_token: str, playlist_id: str, uris: list[str]) -> None:
    """Add tracks to a playlist, chunked at 100 URIs per call (API limit)."""
    for i in range(0, len(uris), 100):
        chunk = uris[i:i + 100]
        resp = requests.post(
            f"{API_BASE}/playlists/{playlist_id}/items",
            headers={**_auth_headers(access_token), "Content-Type": "application/json"},
            json={"uris": chunk},
        )
        _raise_with_detail(resp)


def resolve_playlist(access_token: str, playlist: list[dict], max_workers: int = 10) -> dict:
    """Resolve EVERY track in a playlist against Spotify concurrently, so the
    UI can show album art and a confirmed URI for each -- and so tracks that
    aren't on Spotify are dropped before the user ever sees them.

    Runs the ~N searches in a thread pool (same lesson as discovery.py:
    sequential per-track network calls are painfully slow). Returns:
        {
          "resolved": [ {track dict + uri + image_url + album}, ... ],
          "dropped":  [ {original track dict}, ... ]   # not found on Spotify
        }
    Input tracks are dicts with at least track_name, artist_name, source.
    """
    def _resolve_one(track):
        try:
            info = resolve_track(access_token, track["track_name"], track["artist_name"])
        except Exception:
            info = None
        return track, info

    resolved = []
    dropped = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for track, info in executor.map(_resolve_one, playlist):
            if info is None:
                dropped.append(track)
            else:
                enriched = dict(track)
                enriched["track_uri"] = info["uri"]
                enriched["image_url"] = info["image_url"]
                enriched["album"] = info["album"]
                resolved.append(enriched)

    return {"resolved": resolved, "dropped": dropped}


def push_playlist(
    access_token: str,
    playlist: list[dict],
    name: str,
    description: str = "",
    public: bool = False,
) -> dict:
    """Push a blended playlist (from playlist.py's assemble_playlist) to
    Spotify. Library tracks use their existing track_uri directly;
    discovery tracks are resolved via search first.

    Returns {'playlist_url': str, 'added_count': int, 'skipped': list[dict]}
    -- skipped tracks are ones we couldn't find a confident Spotify match
    for, reported rather than silently dropped.
    """
    uris = []
    skipped = []

    for track in playlist:
        if track.get("track_uri"):
            uris.append(track["track_uri"])
        else:
            resolved = search_track(access_token, track["track_name"], track["artist_name"])
            if resolved:
                uris.append(resolved)
            else:
                skipped.append(track)

    created = create_playlist(access_token, name, description, public)
    playlist_id = created["id"]

    if uris:
        add_tracks_to_playlist(access_token, playlist_id, uris)

    return {
        "playlist_url": created.get("external_urls", {}).get("spotify", ""),
        "added_count": len(uris),
        "skipped": skipped,
    }