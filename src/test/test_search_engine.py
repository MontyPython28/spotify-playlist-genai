"""
Quick demo: run a few example searches against your REAL database to see
search_engine.py working on your actual 15,084 tracks.

This is separate from search_engine.py (a library module with no
runnable output of its own) and from the future agent.py (which will
use Claude to fill in these parameters automatically from a freeform
prompt). This script just lets you sanity-check the search layer itself
first, with parameters set by hand.

Run from the project root:
    python test_search_engine.py
"""

from src.agent.search_engine import search_tracks, SearchParams


def show_results(title: str, results: list[dict]) -> None:
    print(f"\n=== {title} ===")
    if not results:
        print("  (no matches)")
        return
    for r in results[:10]:
        match_info = f" [matched {r['_match_count']} tags]" if "_match_count" in r else ""
        print(f"  {r['track_name']} - {r['artist_name']} "
              f"({r['genre']}, {r['energy']} energy, {r['mood_valence']}){match_info}")


def main():
    show_results(
        "Victory parade",
        search_tracks(SearchParams(
            tags=["triumphant", "celebration", "victory", "anthemic", "upbeat"],
            energy_min="medium-high",
        ))
    )

    show_results(
        "Studying / focus",
        search_tracks(SearchParams(
            tags=["studying", "focused", "calm"],
            pace_max="relaxed",
        ))
    )

    show_results(
        "My favorites (most played)",
        search_tracks(SearchParams(
            sort_by="play_count",
            limit=10,
        ))
    )

    show_results(
        "Sad but not heavy (excluding aggressive)",
        search_tracks(SearchParams(
            valence_max="negative",
            exclude_tags=["aggressive", "party"],
        ))
    )

    show_results(
        "Haven't heard in a long time",
        search_tracks(SearchParams(
            min_days_since_last_played=365,
            sort_by="recent",
            limit=10,
        ))
    )


if __name__ == "__main__":
    main()