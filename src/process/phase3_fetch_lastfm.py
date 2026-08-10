"""
Phase 3: Fetch Last.fm grounding data (community tags + real duration)
for every unique track in your library.

Requests are paced to start at most ~5/sec (Last.fm's documented
guidance), but since they run concurrently, completions happen much
faster than the old fully-serial version. Checkpointed every 200
tracks (thread-safe) so it can be safely interrupted and resumed.

Run from the project root:
    python src/agent/phase3_fetch_lastfm.py

Resume after interruption (auto-detects checkpoint):
    python src/agent/phase3_fetch_lastfm.py
"""

import os
import json
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from dotenv import load_dotenv

STATS_FILE = Path("data/processed/track_stats.parquet")
OUTPUT_FILE = Path("data/processed/lastfm_grounding.json")
CHECKPOINT_EVERY = 200

LASTFM_BASE_URL = "http://ws.audioscrobbler.com/2.0/"
MAX_WORKERS = 10          # backed off from 20 -- a burst of many simultaneous
                           # DNS lookups at once may have contributed to the
                           # resolution failures observed; still much faster
                           # than the original fully-serial version.
MAX_REQUESTS_PER_SEC = 5  # Last.fm's documented guidance -- we gate call
                           # STARTS to this rate; completions overlap
                           # since calls run concurrently, so real
                           # throughput approaches this rate rather than
                           # being limited by per-call latency.
MAX_RETRIES = 3            # raised from 2, with longer backoff below --
                            # DNS/connection errors are often transient.
RETRY_DELAY = 8

_checkpoint_lock = threading.Lock()

# Shared session with connection pooling -- reuses persistent TCP/TLS
# connections instead of re-handshaking on every single call, which was
# almost certainly the real cause of the ~7.5s/call latency observed
# (versus the ~1-1.5s expected for a plain HTTP round trip).
_session = requests.Session()
_adapter = HTTPAdapter(pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)


class RateLimiter:
    """Gates call starts to at most `rate` per second, shared across threads."""
    def __init__(self, rate: float):
        self.min_interval = 1.0 / rate
        self.lock = threading.Lock()
        self.last_call = 0.0

    def acquire(self):
        with self.lock:
            now = time.monotonic()
            wait = self.last_call + self.min_interval - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self.last_call = now


rate_limiter = RateLimiter(MAX_REQUESTS_PER_SEC)


def is_unresolved(result: dict) -> bool:
    """A track counts as 'not yet successfully fetched' if we have no tags
    AND no duration -- this covers both tracks we haven't tried yet and
    tracks where a past attempt failed (network error, DNS blip, etc.) but
    got recorded anyway. Treating these as retriable means a failed fetch
    is never silently permanent -- just re-running the script will pick
    them back up. The rare true case of "track exists on Last.fm but has
    genuinely zero tags and zero duration" gets retried too, which just
    costs a wasted call, not a correctness problem."""
    return not result.get("tags") and result.get("duration_sec") is None


def load_checkpoint() -> dict:
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            data = json.load(f)
        print(f"Resuming: {len(data):,} tracks already fetched.")
        return data
    return {}


def save_checkpoint(data: dict) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    tmp.replace(OUTPUT_FILE)


def fetch_track_grounding(api_key: str, artist: str, track: str) -> dict:
    """Single call: track.getInfo includes both duration and embedded toptags.
    Uses the shared, connection-pooled session (see module setup above)."""
    for attempt in range(MAX_RETRIES):
        try:
            rate_limiter.acquire()
            params = {
                "method": "track.getInfo", "artist": artist, "track": track,
                "api_key": api_key, "format": "json", "autocorrect": 1,
            }
            resp = _session.get(LASTFM_BASE_URL, params=params, timeout=10).json()
            track_data = resp.get("track", {})

            duration_ms = track_data.get("duration")
            duration_sec = (
                int(duration_ms) // 1000 if duration_ms and int(duration_ms) > 0 else None
            )

            raw_tags = track_data.get("toptags", {}).get("tag", [])
            tag_names = [t["name"] for t in raw_tags[:8]] if raw_tags else []

            return {"tags": tag_names, "duration_sec": duration_sec}
        except (requests.RequestException, ValueError) as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"  Failed after {MAX_RETRIES} attempts: {artist} - {track}: {e}")
                return {"tags": [], "duration_sec": None}
    return {"tags": [], "duration_sec": None}


def main():
    load_dotenv()
    api_key = os.getenv("LASTFM_API_KEY")
    if not api_key:
        raise EnvironmentError("LASTFM_API_KEY not found in .env")

    df = pd.read_parquet(STATS_FILE)
    print(f"Loaded {len(df):,} tracks.")

    grounding = load_checkpoint()
    pending = df[df["track_uri"].apply(
        lambda u: u not in grounding or is_unresolved(grounding[u])
    )]

    if len(pending) == 0:
        print("All tracks already have grounding data. Nothing to do.")
        return

    total = len(df)
    already_done = total - len(pending)
    print(f"Fetching grounding for {len(pending):,} tracks ({already_done:,} cached)...")
    print(f"Estimated time: ~{len(pending) / MAX_REQUESTS_PER_SEC / 60:.0f} minutes "
          f"(at ~{MAX_REQUESTS_PER_SEC} req/sec, {MAX_WORKERS} concurrent workers)\n")

    completed = 0
    pending_rows = list(pending.iterrows())

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_uri = {
            executor.submit(fetch_track_grounding, api_key, row["artist_name"], row["track_name"]): row["track_uri"]
            for _, row in pending_rows
        }

        for future in as_completed(future_to_uri):
            uri = future_to_uri[future]
            result = future.result()

            with _checkpoint_lock:
                grounding[uri] = result
                completed += 1

                if completed % CHECKPOINT_EVERY == 0 or completed == len(pending_rows):
                    save_checkpoint(grounding)
                    pct = (already_done + completed) / total * 100
                    found = sum(1 for v in grounding.values() if v["tags"])
                    print(f"  {already_done + completed:,}/{total:,} ({pct:.0f}%) -- "
                          f"{found:,} tracks with real tag data so far -- checkpoint saved")

    save_checkpoint(grounding)
    found = sum(1 for v in grounding.values() if v["tags"])
    print(f"\nDone. {len(grounding):,} tracks processed.")
    print(f"Tracks with real Last.fm tags: {found:,} ({found/len(grounding):.1%})")
    print(f"Tracks with no tags found: {len(grounding) - found:,}")


if __name__ == "__main__":
    main()