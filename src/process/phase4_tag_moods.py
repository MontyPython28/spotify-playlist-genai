"""
Phase 4 v2: Grounded mood tagging via Claude, submitted in sequential
chunks (not one giant batch).

Each chunk (~1,200 tracks, ~$1) is submitted, then this script waits
for it to fully complete and saves its results BEFORE submitting the
next chunk. This means if something goes wrong (credits run out,
billing issue, etc.) partway through, you only ever have ONE chunk's
cost at risk -- every previously completed chunk's results are already
safely written to disk, and any chunks not yet submitted never cost you
anything at all.

Requires phase3_fetch_lastfm.py to have been run first (uses its
output as grounding context).

Run from the project root:
    python src/agent/phase4_grounded_batch.py run

This can be safely interrupted (Ctrl+C) and resumed -- it picks up
exactly where it left off, never re-submitting a chunk that already
succeeded.
"""

import os
import sys
import json
import time
from pathlib import Path

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

STATS_FILE = Path("data/processed/track_stats.parquet")
GROUNDING_FILE = Path("data/processed/lastfm_grounding.json")
OUTPUT_FILE = Path("data/processed/track_stats_tagged.parquet")
STATE_FILE = Path("data/processed/grounded_batch_state.json")

MODEL = "claude-haiku-4-5-20251001"
CHUNK_SIZE = 1200          # ~$1 per chunk at measured per-track cost
POLL_INTERVAL_SEC = 30     # how often to check if the in-flight chunk is done

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
7. If no community tags were provided, rely on artist style and track title, and be \
appropriately more conservative/generic since you have less grounding.

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


def extract_track_id(uri: str) -> str:
    return uri.split(":")[-1]


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"chunks": []}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    tmp.replace(STATE_FILE)


def build_chunks(df: pd.DataFrame) -> list[list[str]]:
    """Split all track_uris into fixed-size chunks (deterministic order)."""
    uris = df["track_uri"].tolist()
    return [uris[i:i + CHUNK_SIZE] for i in range(0, len(uris), CHUNK_SIZE)]


def build_batch_requests(df: pd.DataFrame, track_uris: list[str], grounding: dict) -> list[dict]:
    subset = df[df["track_uri"].isin(track_uris)]
    requests = []
    for _, row in subset.iterrows():
        g = grounding.get(row["track_uri"], {"tags": [], "duration_sec": None})
        user_content = USER_PROMPT_TEMPLATE.format(
            track_name=row["track_name"],
            artist_name=row["artist_name"],
            lastfm_tags=", ".join(g["tags"]) if g["tags"] else "none found",
            duration_sec=g["duration_sec"] or "unknown",
            play_count=row["play_count"],
            skip_rate=row["skip_rate"],
        )
        requests.append({
            "custom_id": extract_track_id(row["track_uri"]),
            "params": {
                "model": MODEL,
                "max_tokens": 600,
                "temperature": 0,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_content}],
            },
        })
    return requests


def append_results_to_output(parsed: dict) -> None:
    """Merge newly tagged tracks into the running output parquet."""
    if OUTPUT_FILE.exists():
        df = pd.read_parquet(OUTPUT_FILE)
    else:
        df = pd.read_parquet(STATS_FILE)
        for col in ["genre", "mood_valence", "mood_arousal", "mood_tags", "energy",
                    "perceived_pace", "sound_tags", "lyrical_themes", "situational_tags",
                    "avoid_if", "track_recognized", "vibe_summary"]:
            df[col] = None

    def get(uri, *path):
        track_id = extract_track_id(uri)
        val = parsed.get(track_id)
        if val is None:
            return None
        for p in path:
            val = val.get(p) if isinstance(val, dict) else None
            if val is None:
                return None
        return val

    mask = df["track_uri"].apply(lambda u: extract_track_id(u) in parsed)
    df.loc[mask, "genre"] = df.loc[mask, "track_uri"].apply(lambda u: get(u, "genre"))
    df.loc[mask, "mood_valence"] = df.loc[mask, "track_uri"].apply(lambda u: get(u, "mood", "valence"))
    df.loc[mask, "mood_arousal"] = df.loc[mask, "track_uri"].apply(lambda u: get(u, "mood", "arousal"))
    df.loc[mask, "mood_tags"] = df.loc[mask, "track_uri"].apply(lambda u: get(u, "mood", "tags"))
    df.loc[mask, "energy"] = df.loc[mask, "track_uri"].apply(lambda u: get(u, "energy"))
    df.loc[mask, "perceived_pace"] = df.loc[mask, "track_uri"].apply(lambda u: get(u, "perceived_pace"))
    df.loc[mask, "sound_tags"] = df.loc[mask, "track_uri"].apply(lambda u: get(u, "sound_tags"))
    df.loc[mask, "lyrical_themes"] = df.loc[mask, "track_uri"].apply(lambda u: get(u, "lyrical_themes"))
    df.loc[mask, "situational_tags"] = df.loc[mask, "track_uri"].apply(lambda u: get(u, "situational_tags"))
    df.loc[mask, "avoid_if"] = df.loc[mask, "track_uri"].apply(lambda u: get(u, "avoid_if"))
    df.loc[mask, "track_recognized"] = df.loc[mask, "track_uri"].apply(lambda u: get(u, "track_recognized"))
    df.loc[mask, "vibe_summary"] = df.loc[mask, "track_uri"].apply(lambda u: get(u, "vibe_summary"))

    df.to_parquet(OUTPUT_FILE, index=False)


def run(client: Anthropic, df: pd.DataFrame, grounding: dict) -> None:
    chunks = build_chunks(df)
    state = load_state()

    while len(state["chunks"]) < len(chunks):
        idx = len(state["chunks"])
        state["chunks"].append({
            "chunk_index": idx,
            "track_uris": chunks[idx],
            "batch_id": None,
            "status": "pending",  # pending -> submitted -> ended -> saved
        })
    save_state(state)

    total_chunks = len(chunks)
    print(f"Total chunks: {total_chunks} (~{CHUNK_SIZE} tracks each)")

    for chunk_state in state["chunks"]:
        idx = chunk_state["chunk_index"]

        if chunk_state["status"] == "saved":
            continue

        print(f"\n--- Chunk {idx + 1}/{total_chunks} ---")

        if chunk_state["status"] == "pending":
            requests = build_batch_requests(df, chunk_state["track_uris"], grounding)
            batch = client.messages.batches.create(requests=requests)
            chunk_state["batch_id"] = batch.id
            chunk_state["status"] = "submitted"
            save_state(state)
            print(f"Submitted batch {batch.id} ({len(requests)} tracks)")

        while chunk_state["status"] == "submitted":
            batch = client.messages.batches.retrieve(chunk_state["batch_id"])
            print(f"  Status: {batch.processing_status} | counts: {batch.request_counts}")
            if batch.processing_status == "ended":
                chunk_state["status"] = "ended"
                save_state(state)
                break
            time.sleep(POLL_INTERVAL_SEC)

        if chunk_state["status"] == "ended":
            results = client.messages.batches.results(chunk_state["batch_id"])
            parsed = {}
            errors = 0
            for entry in results:
                if entry.result.type != "succeeded":
                    errors += 1
                    continue
                try:
                    text = entry.result.message.content[0].text
                    parsed[entry.custom_id] = json.loads(strip_markdown_fences(text))
                except (json.JSONDecodeError, IndexError, AttributeError):
                    errors += 1

            append_results_to_output(parsed)
            chunk_state["status"] = "saved"
            save_state(state)
            print(f"  Saved {len(parsed)} tagged tracks ({errors} errors) to {OUTPUT_FILE}")

    print(f"\nAll {total_chunks} chunks complete. Final output: {OUTPUT_FILE}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] != "run":
        print("Usage: python phase4_grounded_batch.py run")
        sys.exit(1)

    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not found in .env")

    if not GROUNDING_FILE.exists():
        raise FileNotFoundError(
            f"{GROUNDING_FILE} not found. Run phase3_fetch_lastfm.py first."
        )

    client = Anthropic(api_key=api_key)
    df = pd.read_parquet(STATS_FILE)
    with open(GROUNDING_FILE, encoding="utf-8") as f:
        grounding = json.load(f)

    print(f"Loaded {len(df):,} tracks, {len(grounding):,} with grounding data.")
    run(client, df, grounding)


if __name__ == "__main__":
    main()