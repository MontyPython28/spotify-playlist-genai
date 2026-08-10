"""
Phase 2 (fixed): Aggregate raw streaming events into per-track stats.

For display metadata (track_name/artist_name/album_name), we take the
most common (mode) value seen for that track_uri, since these should be
consistent except for the rare album_name variation that caused the bug.
Note: groups by track_uri ALONE, not track_uri + track_name + artist_name +
album_name

Run from the project root:
    python src/agent/phase2_aggregate.py
"""

from pathlib import Path

import pandas as pd

INPUT_FILE = Path("data/processed/streaming_history_clean.parquet")
OUTPUT_FILE = Path("data/processed/track_stats.parquet")

MIN_MS_FOR_REAL_LISTEN = 30_000


def load_clean_history(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} clean play events.")
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["is_real_listen"] = (
        (df["ms_played"] >= MIN_MS_FOR_REAL_LISTEN) & (~df["skipped"])
    )
    df["hour_of_day"] = (df["ts"].dt.hour + 8) % 24
    df["is_intentional"] = df["reason_start"].isin(["clickrow", "trackdone", "playbtn"])
    return df


def _mode_or_first(series: pd.Series):
    """Most common value in the group; falls back to the first value if
    there's a tie or the mode computation is empty for some reason."""
    m = series.mode()
    return m.iloc[0] if len(m) > 0 else series.iloc[0]


def aggregate_per_track(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse play events into one row per unique track_uri."""
    today = pd.Timestamp.now(tz="UTC")

    agg = (
        df.groupby("track_uri")
        .agg(
            track_name=("track_name", _mode_or_first),
            artist_name=("artist_name", _mode_or_first),
            album_name=("album_name", _mode_or_first),
            play_count=("ts", "count"),
            real_listen_count=("is_real_listen", "sum"),
            skip_count=("skipped", "sum"),
            total_ms_played=("ms_played", "sum"),
            avg_ms_played=("ms_played", "mean"),
            first_played=("ts", "min"),
            last_played=("ts", "max"),
            intentional_play_count=("is_intentional", "sum"),
        )
        .reset_index()
    )

    agg["skip_rate"] = (agg["skip_count"] / agg["play_count"]).round(3)
    agg["real_listen_rate"] = (agg["real_listen_count"] / agg["play_count"]).round(3)
    agg["intentional_rate"] = (agg["intentional_play_count"] / agg["play_count"]).round(3)
    agg["days_since_last_played"] = (today - agg["last_played"]).dt.days
    agg["days_in_library"] = (today - agg["first_played"]).dt.days
    agg["completion_rate"] = None

    return agg


def add_peak_hour(df_events: pd.DataFrame, df_agg: pd.DataFrame) -> pd.DataFrame:
    peak_hour = (
        df_events.groupby("track_uri")["hour_of_day"]
        .agg(lambda x: x.mode().iloc[0])
        .rename("peak_hour")
        .reset_index()
    )
    return df_agg.merge(peak_hour, on="track_uri", how="left")


def print_summary(df: pd.DataFrame) -> None:
    print(f"\nAggregated to {len(df):,} unique tracks.")
    dupes_check = df["track_uri"].duplicated().sum()
    print(f"Duplicate track_uri check: {dupes_check} (should be 0)")
    print(f"Date range: {df['first_played'].min()} to {df['last_played'].max()}")

    print("\n--- Listening behaviour overview ---")
    print(f"Median plays per track:       {df['play_count'].median():.0f}")
    print(f"Max plays for one track:      {df['play_count'].max():,}  "
          f"({df.loc[df['play_count'].idxmax(), 'track_name']} - "
          f"{df.loc[df['play_count'].idxmax(), 'artist_name']})")
    print(f"Median skip rate:             {df['skip_rate'].median():.1%}")

    print("\n--- Your most played tracks ---")
    top10 = df.nlargest(10, "play_count")[
        ["track_name", "artist_name", "play_count", "skip_rate", "real_listen_rate"]
    ]
    print(top10.to_string(index=False))


def main():
    df_events = load_clean_history(INPUT_FILE)
    df_events = add_derived_columns(df_events)

    df_agg = aggregate_per_track(df_events)
    df_agg = add_peak_hour(df_events, df_agg)

    del df_events

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_agg.to_parquet(OUTPUT_FILE, index=False)

    print_summary(df_agg)
    print(f"\nSaved track stats to {OUTPUT_FILE}")
    print(f"Shape: {df_agg.shape}")


if __name__ == "__main__":
    main()