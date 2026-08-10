"""
Quick test: fetch real Last.fm tags for 2 tracks to see what crowd-sourced
grounding data actually looks like, before designing the grounded prompt.

Run from the project root:
    python test_lastfm_tags.py
"""

import os

import requests
from dotenv import load_dotenv

LASTFM_BASE_URL = "http://ws.audioscrobbler.com/2.0/"

TEST_TRACKS = [
    {"track_name": "London Boy", "artist_name": "Taylor Swift"},
    {"track_name": "cardigan", "artist_name": "Taylor Swift"},
]


def get_top_tags(api_key: str, artist: str, track: str) -> list[dict]:
    """Fetch community tags for a track, ordered by tag count (popularity)."""
    params = {
        "method": "track.getTopTags",
        "artist": artist,
        "track": track,
        "api_key": api_key,
        "format": "json",
        "autocorrect": 1,  # let Last.fm fix minor spelling/formatting mismatches
    }
    resp = requests.get(LASTFM_BASE_URL, params=params, timeout=10)
    data = resp.json()

    if "error" in data:
        print(f"  Last.fm error: {data.get('message', 'unknown error')}")
        return []

    tags = data.get("toptags", {}).get("tag", [])
    return tags


def get_track_info(api_key: str, artist: str, track: str) -> dict:
    """Fetch general track info including play/listener counts and duration."""
    params = {
        "method": "track.getInfo",
        "artist": artist,
        "track": track,
        "api_key": api_key,
        "format": "json",
        "autocorrect": 1,
    }
    resp = requests.get(LASTFM_BASE_URL, params=params, timeout=10)
    data = resp.json()

    if "error" in data:
        return {}

    return data.get("track", {})


def main():
    load_dotenv()
    api_key = os.getenv("LASTFM_API_KEY")
    if not api_key:
        raise EnvironmentError("LASTFM_API_KEY not found in .env")

    for track in TEST_TRACKS:
        print(f"=== {track['track_name']} - {track['artist_name']} ===")

        tags = get_top_tags(api_key, track["artist_name"], track["track_name"])
        if tags:
            print("Top community tags (by tag count):")
            for tag in tags[:10]:
                print(f"  - {tag['name']} (count: {tag['count']})")
        else:
            print("No tags found.")

        info = get_track_info(api_key, track["artist_name"], track["track_name"])
        if info:
            duration_ms = info.get("duration")
            listeners = info.get("listeners")
            playcount = info.get("playcount")
            print(f"Duration (ms): {duration_ms}")
            print(f"Global listeners: {listeners}")
            print(f"Global playcount: {playcount}")

        print()


if __name__ == "__main__":
    main()