"""
Phase 1: Load and clean Spotify Extended Streaming History.

Reads all Streaming_History_Audio_*.json files from raw_data/,
keeps only actual music track plays (drops podcasts/audiobooks/video),
strips PII (ip address), and saves a clean intermediate dataset
to data/processed/streaming_history_clean.parquet

Memory note: this processes one raw JSON file at a time (load -> clean
-> discard the raw records) rather than loading every year's raw JSON
into memory simultaneously before combining. For an export this size
(~120MB raw JSON total) either approach works fine on a normal machine
-- loading everything at once would peak somewhere around a few hundred
MB -- but doing it file-by-file caps peak memory at roughly one file's
raw JSON plus the running cleaned (smaller) result, which scales better
if you ever point this at a much larger export.

Run from the project root:
    python src/agent/phase1_load_data.py
"""

import glob
import json
from pathlib import Path

import pandas as pd

RAW_DATA_DIR = Path("raw_data")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "streaming_history_clean.parquet"


def clean_to_music_only(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to actual song plays and drop unneeded / sensitive columns."""

    # Keep only rows that are actual music tracks.
    # Podcast episodes / audiobooks have null track_name; video history
    # is in a separate file family we never load (different filename
    # pattern: Streaming_History_Video_*.json).
    music_mask = df["master_metadata_track_name"].notna() & df[
        "spotify_track_uri"
    ].notna()
    df = df[music_mask].copy()

    # Columns we don't need at all for this project.
    drop_cols = [
        "ip_addr",  # PII, not needed
        "episode_name",
        "episode_show_name",
        "spotify_episode_uri",
        "audiobook_title",
        "audiobook_uri",
        "audiobook_chapter_uri",
        "audiobook_chapter_title",
        "offline_timestamp",  # redundant with ts
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Rename to friendlier, shorter names for downstream work.
    df = df.rename(
        columns={
            "master_metadata_track_name": "track_name",
            "master_metadata_album_artist_name": "artist_name",
            "master_metadata_album_album_name": "album_name",
            "spotify_track_uri": "track_uri",
        }
    )

    # Parse timestamp properly (UTC ISO8601 -> pandas datetime).
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    # Sanity bound on ms_played: drop negative/nonsensical values.
    # We don't cap the high end here (some long ms_played values are
    # legitimate looped/long tracks) -- aggregation in Phase 2 will
    # handle outliers more carefully with real stats instead of a
    # guessed cutoff.
    df = df[df["ms_played"] >= 0]

    df = df.reset_index(drop=True)
    return df


def load_all_audio_history(raw_dir: Path) -> pd.DataFrame:
    """Load and clean every Streaming_History_Audio_*.json file.

    Processes one file at a time: load raw JSON -> clean immediately ->
    keep only the cleaned (smaller) result, then move to the next file.
    The raw records/DataFrame for a given file are eligible for garbage
    collection as soon as that file's cleaning step finishes, instead of
    staying in memory until every file has been read.
    """
    pattern = str(raw_dir / "Streaming_History_Audio_*.json")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No files matched {pattern}. Check that raw_data/ contains "
            "the Streaming_History_Audio_*.json files from your Spotify export."
        )

    print(f"Found {len(files)} audio history files:")
    for f in files:
        print(f"  - {f}")
    print()

    cleaned_frames = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            records = json.load(fh)
        raw_df = pd.DataFrame(records)
        raw_count = len(raw_df)

        cleaned = clean_to_music_only(raw_df)
        cleaned_frames.append(cleaned)

        print(f"  {f}: {raw_count:,} raw events -> {len(cleaned):,} music rows kept")

        # raw_df and records aren't referenced again after this point in
        # the loop, so they're eligible for garbage collection before we
        # load the next file.

    df = pd.concat(cleaned_frames, ignore_index=True)
    print(f"\nLoaded and cleaned {len(df):,} total music events across all years.")
    return df


def main():
    df_clean = load_all_audio_history(RAW_DATA_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(OUTPUT_FILE, index=False)

    print(f"\nSaved clean dataset to {OUTPUT_FILE}")
    print(f"Final shape: {df_clean.shape}")
    print(f"Date range: {df_clean['ts'].min()} to {df_clean['ts'].max()}")
    print(f"Unique tracks: {df_clean['track_uri'].nunique():,}")
    print(f"Unique artists: {df_clean['artist_name'].nunique():,}")
    print("\nSample row:")
    print(df_clean.iloc[0])


if __name__ == "__main__":
    main()