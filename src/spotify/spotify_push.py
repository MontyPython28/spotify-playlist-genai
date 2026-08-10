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