"""
Phase 5: Load the tagged track dataset into SQLite.

List fields (mood_tags, situational_tags, sound_tags, lyrical_themes,
avoid_if) are stored as JSON-encoded text, since SQLite has no native
array/list column type -- this is a standard, well-supported pattern.
Use json.loads() when reading them back out in Python.

Creates indexes on columns the recommendation agent will filter on
often (genre, energy, mood_valence, play_count) for fast queries later.

Run from the project root:
    python src/agent/phase5_load_sqlite.py
"""

import json
import sqlite3
from pathlib import Path

import pandas as pd

INPUT_FILE = Path("data/processed/track_stats_tagged.parquet")
DB_FILE = Path("data/mood_agent.db")

LIST_COLUMNS = ["mood_tags", "sound_tags", "lyrical_themes", "situational_tags", "avoid_if"]


def json_encode_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in LIST_COLUMNS:
        df[col] = df[col].apply(
            lambda x: json.dumps(x.tolist() if hasattr(x, "tolist") else x)
            if x is not None else json.dumps([])
        )
    return df


def main():
    df = pd.read_parquet(INPUT_FILE)
    print(f"Loaded {len(df):,} tracks from {INPUT_FILE}")

    tagged = df[df["genre"].notna()].copy()
    dropped = len(df) - len(tagged)
    if dropped:
        print(f"Excluding {dropped} untagged track(s) from the database.")

    tagged = json_encode_list_columns(tagged)

    tagged["track_recognized"] = tagged["track_recognized"].apply(
        lambda x: int(bool(x)) if pd.notna(x) else None
    )

    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)

    tagged.to_sql("tracks", conn, if_exists="replace", index=False)

    cursor = conn.cursor()
    for col in ["genre", "energy", "mood_valence", "mood_arousal", "play_count", "track_uri"]:
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{col} ON tracks({col})")
    conn.commit()

    count = cursor.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
    print(f"\nLoaded {count:,} tracks into {DB_FILE}")

    print("\nSample query: high-energy positive tracks")
    sample = cursor.execute("""
        SELECT track_name, artist_name, genre, energy, mood_valence
        FROM tracks
        WHERE energy IN ('high', 'very-high') AND mood_valence IN ('positive', 'very-positive')
        ORDER BY play_count DESC
        LIMIT 5
    """).fetchall()
    for row in sample:
        print(f"  {row}")

    conn.close()
    print(f"\nDatabase ready at {DB_FILE}")
    print("List columns (mood_tags, situational_tags, etc.) are stored as JSON text --")
    print("use json.loads() when reading them back in Python, or SQLite's json_each()")
    print("in SQL queries if you want to filter inside them directly.")


if __name__ == "__main__":
    main()