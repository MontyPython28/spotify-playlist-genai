"""
search_engine.py -- the safe, hand-written query layer for the mood agent.

This is the ONE place SQL gets written for track search. Claude never
writes SQL directly; it only fills in the parameters of search_tracks(),
which then builds a parameterized, injection-safe query using this
tested logic. See agent.py for the layer that translates a freeform
user prompt into these parameters.

Schema reminder (columns in the `tracks` table):
  Scalar:  track_uri, track_name, artist_name, album_name, genre,
           energy, mood_valence, mood_arousal, perceived_pace,
           play_count, skip_rate, days_since_last_played, track_recognized
  JSON text (use json.loads() after fetching, or json_each() in SQL):
           mood_tags, sound_tags, lyrical_themes, situational_tags, avoid_if

NOTE: there is no BPM column -- Claude's BPM guesses were dropped for
quality reasons and Last.fm's free tier doesn't provide it either.
perceived_pace is the closest proxy available today.
"""

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

DB_FILE = Path("data/mood_agent.db")

ENERGY_LEVELS = ["very-low", "low", "medium-low", "medium", "medium-high", "high", "very-high"]
VALENCE_LEVELS = ["very-negative", "negative", "neutral", "positive", "very-positive"]
PACE_LEVELS = ["very-slow", "slow", "relaxed", "walking", "moderate", "driving", "fast", "frantic"]

TAG_COLUMNS = ["mood_tags", "situational_tags", "sound_tags", "lyrical_themes"]


def _expand_range(levels: list[str], min_val: str | None, max_val: str | None) -> list[str]:
    """Turn a min/max ordinal range into the explicit list of allowed values.
    e.g. levels=ENERGY_LEVELS, min_val='medium-high', max_val=None
         -> ['medium-high', 'high', 'very-high']"""
    lo = levels.index(min_val) if min_val else 0
    hi = levels.index(max_val) if max_val else len(levels) - 1
    if lo > hi:
        lo, hi = hi, lo
    return levels[lo:hi + 1]


@dataclass
class SearchParams:
    """Structured search parameters. Claude fills these in from a freeform
    prompt (see agent.py) -- every field is optional; unset fields impose
    no filter."""
    tags: list[str] = field(default_factory=list)
    exclude_tags: list[str] = field(default_factory=list)
    genres: list[str] = field(default_factory=list)
    exclude_artists: list[str] = field(default_factory=list)
    exclude_genres: list[str] = field(default_factory=list)
    require_lyrics: bool = False  # uses lyrical_themes emptiness (the tagging
                                    # schema's own "empty if instrumental or
                                    # unknown" convention) rather than guessing
                                    # at instrumental-associated genre words --
                                    # generalizes across every genre, not just
                                    # ones enumerated by hand
    energy_min: str | None = None
    energy_max: str | None = None
    valence_min: str | None = None
    valence_max: str | None = None
    pace_min: str | None = None
    pace_max: str | None = None
    min_play_count: int | None = None
    max_play_count: int | None = None
    min_days_since_last_played: int | None = None
    only_recognized: bool = False
    sort_by: str = "relevance"
    limit: int = 30
    exclude_uris: list[str] = field(default_factory=list)  # e.g. exclude the seed track in "songs like X"


def search_tracks(params: SearchParams, db_path: Path = DB_FILE) -> list[dict]:
    """Execute a search against the tracks database. Returns a list of
    track dicts, ranked by relevance (number of matched tags) when
    sort_by='relevance', or by the requested SQL ordering otherwise."""

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    where_clauses = []
    query_params = []

    if params.genres:
        genre_conditions = " OR ".join(["genre LIKE ?" for _ in params.genres])
        where_clauses.append(f"({genre_conditions})")
        query_params.extend([f"%{g}%" for g in params.genres])

    if params.exclude_artists:
        exclude_conditions = " AND ".join(["artist_name NOT LIKE ?" for _ in params.exclude_artists])
        where_clauses.append(f"({exclude_conditions})")
        query_params.extend([f"%{a}%" for a in params.exclude_artists])

    if params.exclude_genres:
        # A real, dedicated mechanism -- previously "no rap"/"not country"
        # style requests had no correct way to be expressed at all:
        # exclude_tags searches mood/sound/situational tags and
        # vibe_summary, but never the genre column, so a genre exclusion
        # put there would silently do nothing.
        exclude_conditions = " AND ".join(["genre NOT LIKE ?" for _ in params.exclude_genres])
        where_clauses.append(f"({exclude_conditions})")
        query_params.extend([f"%{g}%" for g in params.exclude_genres])

    if params.require_lyrics:
        # lyrical_themes is empty specifically when a track is instrumental
        # or unknown (per the tagging schema's own convention) -- this
        # generalizes across every genre, unlike guessing at a fixed list
        # of "instrumental-sounding" genre words.
        where_clauses.append("json_array_length(lyrical_themes) > 0")

    if params.energy_min or params.energy_max:
        allowed = _expand_range(ENERGY_LEVELS, params.energy_min, params.energy_max)
        where_clauses.append(f"energy IN ({','.join(['?']*len(allowed))})")
        query_params.extend(allowed)

    if params.valence_min or params.valence_max:
        allowed = _expand_range(VALENCE_LEVELS, params.valence_min, params.valence_max)
        where_clauses.append(f"mood_valence IN ({','.join(['?']*len(allowed))})")
        query_params.extend(allowed)

    if params.pace_min or params.pace_max:
        allowed = _expand_range(PACE_LEVELS, params.pace_min, params.pace_max)
        where_clauses.append(f"perceived_pace IN ({','.join(['?']*len(allowed))})")
        query_params.extend(allowed)

    if params.min_play_count is not None:
        where_clauses.append("play_count >= ?")
        query_params.append(params.min_play_count)

    if params.max_play_count is not None:
        where_clauses.append("play_count <= ?")
        query_params.append(params.max_play_count)

    if params.min_days_since_last_played is not None:
        where_clauses.append("days_since_last_played >= ?")
        query_params.append(params.min_days_since_last_played)

    if params.only_recognized:
        where_clauses.append("track_recognized = 1")

    if params.exclude_uris:
        placeholders = ",".join(["?"] * len(params.exclude_uris))
        where_clauses.append(f"track_uri NOT IN ({placeholders})")
        query_params.extend(params.exclude_uris)

    if params.tags:
        tag_match_parts = []
        for col in TAG_COLUMNS:
            placeholders = ",".join(["?"] * len(params.tags))
            tag_match_parts.append(
                f"EXISTS (SELECT 1 FROM json_each(tracks.{col}) je WHERE je.value IN ({placeholders}))"
            )
            query_params.extend(params.tags)
        # vibe_summary is free text, not a JSON array -- substring match
        # against it too, since it's less space-constrained than the
        # categorical tag lists and sometimes captures a concept (e.g. the
        # literal word "aggressive") that didn't make it into the tighter
        # mood_tags list.
        vibe_conditions = " OR ".join(["tracks.vibe_summary LIKE ?" for _ in params.tags])
        tag_match_parts.append(f"({vibe_conditions})")
        query_params.extend([f"%{t}%" for t in params.tags])
        where_clauses.append(f"({' OR '.join(tag_match_parts)})")

    if params.exclude_tags:
        # Widened from exact string match to substring LIKE against the raw
        # JSON text: exact equality only caught tracks tagged with the
        # literal string "aggressive", missing variants like "aggressive
        # vocals". This still won't catch true synonyms with no shared
        # substring (e.g. "intense" when excluding "aggressive") -- that's
        # addressed on the prompt side (see agent.py) by asking Claude for
        # a few synonymous exclude_tags rather than relying on one exact word.
        #
        # KNOWN REMAINING LIMITATION (confirmed via real data): the tagger
        # sometimes chose consistently positive/flattering mood framing for
        # genuinely intense tracks (e.g. "triumphant", "defiant" for a
        # distorted-guitar pop-punk anthem) with NO negatively-connoted word
        # anywhere, including vibe_summary -- no string-matching fix, however
        # widened, can find a word that was never written down. sound_tags
        # tend to be more consistent/objective (e.g. "distorted guitars"
        # reliably appears on aggressive-sounding tracks even when mood_tags
        # avoid saying so), which is why the prompt now also asks Claude for
        # sonic-level exclude_tags proxies, not just mood-word synonyms.
        exclude_parts = []
        for col in TAG_COLUMNS + ["avoid_if"]:
            col_conditions = " OR ".join([f"tracks.{col} LIKE ?" for _ in params.exclude_tags])
            exclude_parts.append(f"({col_conditions})")
            query_params.extend([f"%{t}%" for t in params.exclude_tags])
        vibe_exclude_conditions = " OR ".join(["tracks.vibe_summary LIKE ?" for _ in params.exclude_tags])
        exclude_parts.append(f"({vibe_exclude_conditions})")
        query_params.extend([f"%{t}%" for t in params.exclude_tags])
        where_clauses.append(f"NOT ({' OR '.join(exclude_parts)})")

    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

    order_sql = {
        "play_count": "play_count DESC",
        "recent": "days_since_last_played ASC",
        "random": "RANDOM()",
    }.get(params.sort_by, "play_count DESC")

    fetch_limit = params.limit * 5 if params.sort_by == "relevance" else params.limit

    sql = f"""
        SELECT * FROM tracks
        WHERE {where_sql}
        ORDER BY {order_sql}
        LIMIT ?
    """
    query_params.append(fetch_limit)

    rows = cur.execute(sql, query_params).fetchall()
    results = [dict(row) for row in rows]

    for r in results:
        for col in TAG_COLUMNS + ["avoid_if"]:
            r[col] = json.loads(r[col]) if r[col] else []

    if params.sort_by == "relevance" and params.tags:
        tag_set = set(params.tags)
        for r in results:
            all_track_tags = set()
            for col in TAG_COLUMNS:
                all_track_tags.update(r[col])
            r["_match_count"] = len(tag_set & all_track_tags)
        results.sort(key=lambda r: (r["_match_count"], r["play_count"]), reverse=True)

    # De-duplicate by (track_name, artist_name): different track_uris can
    # legitimately point to distinct catalog entries that share a display
    # name (e.g. a standard release vs. a piano/acoustic version vs. a
    # deluxe-edition duplicate) -- technically different tracks, but showing
    # the "same song" several times in one set of results/playlist is a bad
    # experience. Keep the first (best-ranked, given the sort above) instance
    # of each name+artist pair and drop the rest.
    seen = set()
    deduped = []
    for r in results:
        key = (r["track_name"].strip().lower(), r["artist_name"].strip().lower())
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    results = deduped

    conn.close()
    return results[:params.limit]


def find_track(track_name: str, artist_name: str | None = None, db_path=DB_FILE) -> list[dict]:
    """Fuzzy lookup for 'songs like X' -- case-insensitive substring match on
    track name, optionally narrowed by artist. Returns candidates sorted by
    play_count descending (so the user's actual most-played version of a
    track wins if there are near-duplicates), most relevant first."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    if artist_name:
        rows = cur.execute(
            "SELECT * FROM tracks WHERE track_name LIKE ? AND artist_name LIKE ? "
            "ORDER BY play_count DESC LIMIT 5",
            (f"%{track_name}%", f"%{artist_name}%"),
        ).fetchall()
    else:
        rows = cur.execute(
            "SELECT * FROM tracks WHERE track_name LIKE ? ORDER BY play_count DESC LIMIT 5",
            (f"%{track_name}%",),
        ).fetchall()

    results = [dict(row) for row in rows]
    for r in results:
        for col in TAG_COLUMNS + ["avoid_if"]:
            r[col] = json.loads(r[col]) if r[col] else []

    conn.close()
    return results


def similar_track_params(seed: dict, window: int = 1) -> dict:
    """Build search parameters from a seed track's OWN real attributes, for
    'songs like X' queries. Widens energy/valence/pace by `window` levels
    either side of the seed's actual value, since an exact match on a
    single ordinal level would be too restrictive to return good results.
    Returns a plain dict of field values meant to be merged into a
    SearchParams (see agent.py), not a SearchParams itself, so the caller
    can combine it with anything Claude additionally inferred from the
    user's own wording (e.g. "songs like X but calmer")."""

    def widen(levels: list[str], value: str | None) -> tuple[str | None, str | None]:
        if not value or value not in levels:
            return None, None
        idx = levels.index(value)
        lo = max(0, idx - window)
        hi = min(len(levels) - 1, idx + window)
        return levels[lo], levels[hi]

    energy_min, energy_max = widen(ENERGY_LEVELS, seed.get("energy"))
    valence_min, valence_max = widen(VALENCE_LEVELS, seed.get("mood_valence"))
    pace_min, pace_max = widen(PACE_LEVELS, seed.get("perceived_pace"))

    seed_tags = set()
    for col in TAG_COLUMNS:
        seed_tags.update(seed.get(col, []))

    return {
        "tags": list(seed_tags),
        # NOTE: genre is deliberately NOT included as a hard filter here.
        # Locking to the seed's exact genre label caused real false-empty
        # results -- e.g. "similar to cardigan (folk pop) but more upbeat"
        # ANDs a genre that's inherently low-energy-skewed with an explicit
        # higher-energy request, often intersecting to nothing. Mood/sound/
        # situational tag overlap already captures "similar" well; genre
        # stays available for Claude to set explicitly if genre itself is
        # what the user cares about (e.g. "something like X, still rock").
        "energy_min": energy_min, "energy_max": energy_max,
        "valence_min": valence_min, "valence_max": valence_max,
        "pace_min": pace_min, "pace_max": pace_max,
        "exclude_uris": [seed["track_uri"]],
    }