"""
Test the grounded v3 pipeline on lesser-known tracks from your real library
-- specifically a random sample from the "middle" of your play_count
distribution, since that's where Claude's raw recall is least reliable
and grounding should matter most.

Run from the project root:
    python test_phase4_grounded_sample.py
"""

import os
import json

import pandas as pd
import requests
from anthropic import Anthropic
from dotenv import load_dotenv

MODEL = "claude-haiku-4-5-20251001"
LASTFM_BASE_URL = "http://ws.audioscrobbler.com/2.0/"
STATS_FILE = "data/processed/track_stats.parquet"
SAMPLE_SIZE = 6
# "Lesser known" = played a handful of times, not your obsessive top
# tracks (200+ plays) and not one-off noise (played once, might be a
# skip or misclick). This band is where Claude's raw memory is weakest
# and real crowd-sourced grounding should help most.
MIN_PLAYS = 5
MAX_PLAYS = 20

SYSTEM_PROMPT = """You are a music semantic tagging assistant. You will be given a \
track name, artist, real community-sourced tags from Last.fm, and the track's actual \
duration. Use the community tags as ground truth for genre/style -- do not contradict \
them. Your job is to add the layer real tag data can't provide: mood, situational fit, \
lyrical themes, and energy characteristics.

IMPORTANT:
1. Use your own knowledge of the specific track's lyrics/energy/feel whenever you \
recognise it, informed by (not overridden by) the community tags provided.
2. If you don't recognise the specific track, infer cautiously from the community \
tags, artist style, and track title -- and set "track_recognized" to false.
3. Do not assume high energy means happy, or that a sad theme means low energy.
4. Only include "avoid_if" tags when a trait is strongly characteristic -- an empty \
list is expected and normal, do not pad it.
5. Do not invent specific narrative details (e.g. specific locations, specific plot \
details) unless you are genuinely confident they reflect the real song. Prefer \
broader, well-established themes over speculative specifics.
6. Keep every list as SHORT as the schema allows -- use the minimum, not the maximum.

CRITICAL OUTPUT RULE: Respond with ONLY the raw JSON object. No markdown code fences, \
no headers, no commentary, no notes before or after it. Your entire response must be \
parseable as JSON starting from the very first character.

Schema:
{
  "genre": "<primary genre, lowercase -- should align with the community tags given>",
  "mood": {
    "valence": "<very-negative|negative|neutral|positive|very-positive>",
    "arousal": "<very-low|low|medium-low|medium|medium-high|high|very-high>",
    "tags": ["<2-3 specific mood/emotion words, no more>"]
  },
  "energy": "<very-low|low|medium-low|medium|medium-high|high|very-high>",
  "perceived_pace": "<very-slow|slow|relaxed|walking|moderate|driving|fast|frantic>",
  "sound_tags": ["<2-3 tags covering instrumentation/vocals/production>"],
  "lyrical_themes": ["<0-2 themes; empty if instrumental or unknown>"],
  "situational_tags": ["<3-4 tags covering time/place/activity>"],
  "avoid_if": ["<0-2 tags, only if strongly characteristic; empty list expected often>"],
  "track_recognized": <true if you know this specific song, false if inferring>,
  "vibe_summary": "<one sentence, under 15 words>"
}"""

USER_PROMPT_TEMPLATE = """Track: "{track_name}"
Artist: {artist_name}
Community tags (Last.fm, ordered by popularity): {lastfm_tags}
Actual duration: {duration_sec}s
User listening behavior: played {play_count} times, skip rate {skip_rate:.0%}
(Use listening behavior only as context on engagement -- do not let popularity \
change the track's intrinsic musical characteristics.)

Respond with ONLY the JSON object. No other text."""


def strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def get_lastfm_data(api_key: str, artist: str, track: str) -> dict:
    tag_params = {
        "method": "track.getTopTags", "artist": artist, "track": track,
        "api_key": api_key, "format": "json", "autocorrect": 1,
    }
    tag_resp = requests.get(LASTFM_BASE_URL, params=tag_params, timeout=10).json()
    tags = tag_resp.get("toptags", {}).get("tag", [])
    top_tag_names = [t["name"] for t in tags[:8]]

    info_params = {
        "method": "track.getInfo", "artist": artist, "track": track,
        "api_key": api_key, "format": "json", "autocorrect": 1,
    }
    info_resp = requests.get(LASTFM_BASE_URL, params=info_params, timeout=10).json()
    duration_ms = info_resp.get("track", {}).get("duration")

    return {
        "tags": top_tag_names,
        "duration_sec": int(duration_ms) // 1000 if duration_ms and int(duration_ms) > 0 else None,
    }


def sample_lesser_known_tracks() -> list[dict]:
    df = pd.read_parquet(STATS_FILE)
    band = df[(df["play_count"] >= MIN_PLAYS) & (df["play_count"] <= MAX_PLAYS)]
    print(f"Tracks with {MIN_PLAYS}-{MAX_PLAYS} plays: {len(band):,} out of {len(df):,} total")

    sample = band.sample(n=min(SAMPLE_SIZE, len(band)), random_state=None)
    return sample[["track_name", "artist_name", "play_count", "skip_rate"]].to_dict("records")


def main():
    load_dotenv()
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    lastfm_key = os.getenv("LASTFM_API_KEY")
    if not anthropic_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not found in .env")
    if not lastfm_key:
        raise EnvironmentError("LASTFM_API_KEY not found in .env")

    client = Anthropic(api_key=anthropic_key)

    tracks = sample_lesser_known_tracks()
    print(f"\nSampled {len(tracks)} lesser-known tracks:")
    for t in tracks:
        print(f"  - {t['track_name']} - {t['artist_name']} ({t['play_count']} plays)")
    print()

    total_input_tokens = 0
    total_output_tokens = 0

    for track in tracks:
        lastfm = get_lastfm_data(lastfm_key, track["artist_name"], track["track_name"])
        print(f"[Last.fm grounding]: {lastfm['tags']}")

        user_content = USER_PROMPT_TEMPLATE.format(
            track_name=track["track_name"],
            artist_name=track["artist_name"],
            lastfm_tags=", ".join(lastfm["tags"]) if lastfm["tags"] else "none found",
            duration_sec=lastfm["duration_sec"] or "unknown",
            play_count=track["play_count"],
            skip_rate=track["skip_rate"],
        )

        response = client.messages.create(
            model=MODEL,
            max_tokens=600,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )

        text = response.content[0].text
        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

        print(f"=== {track['track_name']} - {track['artist_name']} ===")
        try:
            parsed = json.loads(strip_markdown_fences(text))
            print(json.dumps(parsed, indent=2))
        except json.JSONDecodeError:
            print("WARNING: response was not valid JSON even after stripping fences:")
            print(text)
        print()

    input_cost = total_input_tokens / 1_000_000 * 1.00
    output_cost = total_output_tokens / 1_000_000 * 5.00
    per_track_cost = (input_cost + output_cost) / len(tracks)

    print("--- Token usage & cost (grounded v3, lesser-known sample) ---")
    print(f"Total input tokens:  {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Cost for these {len(tracks)} tracks: ${input_cost + output_cost:.5f}")
    print(f"Estimated cost per track (Batch API, 50% off): ${per_track_cost / 2:.5f}")
    print(f"\nProjected Claude-only cost for 15,284 tracks (Batch API): "
          f"${per_track_cost / 2 * 15284:.2f}")


if __name__ == "__main__":
    main()