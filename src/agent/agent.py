"""
agent.py -- translates a freeform prompt into a search_tracks() call.

Claude never writes SQL. It only fills in the parameters of SearchParams
(see search_engine.py) via tool-calling; the actual query logic is fixed,
tested code.

v2 improvements (after review):
- Vocabulary grounding: real tags from the database are shown to Claude
  so it translates poetic/freeform language into concepts likely to
  actually exist, instead of inventing novel phrasing that silently
  matches nothing.
- A much more detailed system prompt with explicit translation rules
  (e.g. "sad but energetic" != low energy; don't infer genre from mood
  stereotypes; don't apply listening-history filters to ordinary mood
  requests).

Run from the project root:
    python agent.py "songs for a victory parade"
    python agent.py "sad songs but not too heavy"
    python agent.py "something chill for studying"
"""

import os
import sys
import json
import sqlite3
from collections import Counter
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "spotify"))

from search_engine import (
    search_tracks, SearchParams, ENERGY_LEVELS, VALENCE_LEVELS, PACE_LEVELS,
    TAG_COLUMNS, DB_FILE, find_track, similar_track_params,
)
from discovery import discover_new_music
from playlist import assemble_playlist, format_playlist
from spotify_auth import get_valid_access_token
from spotify_push import push_playlist

MODEL = "claude-haiku-4-5-20251001"
VOCAB_SAMPLE_SIZE = 150  # how many of the most common real tags to show Claude
COST_LOG_FILE = Path("data/agent_cost_log.json")


def log_cost(cost_usd: float) -> tuple[float, int]:
    """Append this query's cost to a running session log. Returns
    (all_time_total_cost, all_time_query_count), so repeated use doesn't
    quietly rack up spend unnoticed."""
    COST_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    if COST_LOG_FILE.exists():
        with open(COST_LOG_FILE) as f:
            log = json.load(f)
    else:
        log = {"total_cost_usd": 0.0, "query_count": 0}

    log["total_cost_usd"] += cost_usd
    log["query_count"] += 1

    with open(COST_LOG_FILE, "w") as f:
        json.dump(log, f)

    return log["total_cost_usd"], log["query_count"]


def get_tag_vocabulary(db_path=DB_FILE, top_n: int = VOCAB_SAMPLE_SIZE) -> list[str]:
    """Pull the most common real tag values across all tag columns, so
    Claude can ground generated tags in what actually exists in the
    database instead of inventing plausible-sounding strings that won't
    match anything at query time."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    counter = Counter()
    for col in TAG_COLUMNS:
        rows = cur.execute(f"SELECT {col} FROM tracks WHERE {col} IS NOT NULL").fetchall()
        for (raw,) in rows:
            try:
                tags = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            counter.update(tags)
    conn.close()
    return [tag for tag, _ in counter.most_common(top_n)]


def build_search_tool(vocabulary: list[str]) -> dict:
    vocab_str = ", ".join(sorted(vocabulary))
    return {
        "name": "search_tracks",
        "description": f"""Search the user's tagged music library.

The track database uses this schema:
- genres: primary genre of each track
- valence_min/valence_max: overall emotional positivity/negativity range
- energy_min/energy_max: overall musical energy range
- pace_min/pace_max: how fast the track FEELS (there is no exact BPM data --
  this is the closest available proxy for tempo requests)
- tags: searches across mood tags (e.g. nostalgic, euphoric), situational
  tags (e.g. rainy day, late night, studying, victory parade), sound tags
  (e.g. acoustic guitar, female vocals, synth-heavy), and lyrical themes
  (e.g. heartbreak, nostalgia) all at once -- you don't need to know which
  specific column a concept lives in
- exclude_tags: characteristics to actively avoid, searched across the same
  tag universe as `tags`, negated
- min_play_count / max_play_count / min_days_since_last_played: the user's
  own listening history with a track (favorites, deep cuts, "haven't heard
  in a while") -- NOT for ordinary mood/activity requests

IMPORTANT -- vocabulary grounding: the tags below are a sample of the most
common REAL tags that actually exist in this user's tagged library. When
generating values for `tags` or `exclude_tags`, prefer concepts from this
list, or close variants of them, over inventing novel poetic phrasing --
exact string matching is used, so a tag that doesn't resemble anything in
the real vocabulary will silently match nothing.

Real tag vocabulary sample: {vocab_str}""",
        "input_schema": {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Semantic concepts to match -- mood, situational, sound, lyrical. Prefer real vocabulary shown above.",
                },
                "exclude_tags": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Characteristics to actively exclude, searched across the same tag universe as tags.",
                },
                "genres": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Genre keywords (substring matched). Only use when genre is explicitly requested or clearly important -- don't infer genre from mood stereotypes.",
                },
                "exclude_artists": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Specific artist names to exclude entirely (e.g. 'no Lana Del Rey', 'nothing by Drake'). This is separate from exclude_tags -- use exclude_artists for literal artist names, exclude_tags for mood/sound/situational concepts. Applies to both library search and new-music discovery.",
                },
                "exclude_genres": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Genre keywords to actively exclude (e.g. 'upbeat pop, not rock' -> exclude_genres=['rock']). Separate from exclude_tags -- genre is its own column and is never matched by exclude_tags at all.",
                },
                "require_lyrics": {
                    "type": "boolean",
                    "description": "Set true when the user wants vocal/lyrical tracks specifically, excluding instrumental ones (e.g. 'with lyrics', 'not instrumental'). Checks actual lyrical content in the data rather than relying on genre or tag guessing.",
                },
                "exclude_tracks": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Specific song titles to exclude (e.g. 'not I Wanna Be Your Slave', 'remove Blinding Lights'). Separate from exclude_artists (whole artist) and exclude_tags (mood/sound concepts) -- use this when the user names one specific track they don't want. The title will be matched against the real library, so approximate wording/capitalization is fine.",
                },
                "energy_min": {"type": "string", "enum": ENERGY_LEVELS},
                "energy_max": {"type": "string", "enum": ENERGY_LEVELS},
                "valence_min": {"type": "string", "enum": VALENCE_LEVELS},
                "valence_max": {"type": "string", "enum": VALENCE_LEVELS},
                "pace_min": {"type": "string", "enum": PACE_LEVELS},
                "pace_max": {"type": "string", "enum": PACE_LEVELS},
                "min_play_count": {"type": "integer"},
                "max_play_count": {"type": "integer"},
                "min_days_since_last_played": {"type": "integer"},
                "similar_to_track": {
                    "type": "string",
                    "description": "Set this when the user asks for songs SIMILAR TO a specific named track (e.g. 'songs like Blinding Lights'). The track's own real genre/mood/energy/sound will be looked up and used as the search basis automatically -- you can ALSO set other fields (e.g. energy_max) to adjust the request, e.g. 'songs like X but calmer'.",
                },
                "similar_to_artist": {
                    "type": "string",
                    "description": "Optional artist name to disambiguate similar_to_track if needed (e.g. multiple songs with similar titles).",
                },
                "discover_new": {
                    "type": "boolean",
                    "description": "Set true when the user explicitly wants NEW music they don't already have -- e.g. 'recommend something I haven't heard', 'discover new music like X', 'surprise me with something new'. Do NOT set this for ordinary playlist/mood requests -- only when discovery of unfamiliar music is clearly the intent.",
                },
                "discovery_ratio": {
                    "type": "number",
                    "description": "Fraction of the playlist that should be NEW (discovery) tracks. Follows fixed tiers -- see system prompt for the exact table: 0.3 default, 0.6 for 'some new', 0.9 for plain 'new', 1.0 for 'all new'. Do not set for library-only requests.",
                },
                "sort_by": {"type": "string", "enum": ["relevance", "play_count", "recent", "random"]},
                "limit": {"type": "integer"},
            },
        },
    }


SYSTEM_PROMPT = """You are a music search-query translator. Translate the user's \
freeform music request into a call to the search_tracks tool. You are NOT selecting \
songs yourself and you are NOT writing SQL -- the tool performs the actual search.

## TRANSLATION RULES

Map natural language to the closest available concepts. Examples:
- "happy" -> positive valence and/or positive mood tags
- "really happy / euphoric" -> very-positive valence, possibly higher energy
- "sad" -> negative valence and/or melancholic/sad tags
- "chill" -> lower energy and/or relaxed/slow pace
- "upbeat" -> higher energy and/or positive valence
- "slow" / "driving" / "frantic" -> pace, not energy
- "rainy day", "late night", "studying" -> situational tags
- "piano", "female vocals", "acoustic" -> sound tags
- "heartbreak" -> lyrical theme and/or mood tag
- "rock" -> genre
- "not aggressive" -> exclude_tags

Do not translate words literally without interpreting meaning in context. \
Translate poetic or evocative language into the concepts most likely to exist in the \
real vocabulary shown in the tool description (e.g. "sun through my bedroom window on \
a Sunday morning" -> tags like "sunny", "morning", "peaceful", "relaxed" -- not \
invented phrases like "sunlight" or "bedroom" that won't match anything).

## HARD CONSTRAINTS VS SOFT PREFERENCES

Explicit requirements become search constraints. Explicit exclusions become \
exclude_tags. Soft preferences should stay broad rather than over-restricting the query.

## COMBINED / NUANCED REQUESTS -- DO NOT MAKE UNWARRANTED ASSUMPTIONS

A request can carry several independent constraints. Do not collapse everything into \
one generic interpretation.
- "happy songs for a rainy evening" -> positive valence + rainy/evening situational tags
- "sad but energetic" -> negative valence + HIGHER energy. Do NOT assume sadness implies low energy.
- "happy but calm" -> positive valence + energy_max of medium-low or lower. "Calm" is a \
specific, fairly low point on the energy scale, not just "somewhat less than a typical \
happy song" -- don't leave it at a middling value like medium.
- "victory parade" / "triumphant" / "celebratory" / "anthemic" requests -> energy_min of \
medium-high or higher. These concepts inherently imply high energy (crowds, brass, marching \
bands) -- don't leave energy unset or at a middling value just because the request didn't \
use the literal word "energy".

## TEMPO

There is no exact BPM data. For tempo requests (including specific numbers like "120 \
BPM"), use pace_min/pace_max as the closest proxy, and mention in your reasoning that \
exact BPM filtering isn't available.

## TAGS

Use tags for concepts without their own structured field (mood, situational, sound, and \
lyrical themes are all searched together). Matching is EXACT per tag, not semantic -- a \
track's tag list holds individual short words (e.g. "triumphant", "nostalgic", "rainy \
day"), not full sentences, so:

- Prefer SEVERAL short, atomic tags over one long phrase. "victory parade" as a single \
tag will rarely exact-match anything real -- break it into the underlying concepts \
instead: victory, triumphant, celebration, anthemic.
- For the CORE concept(s) in the request, provide 3-6 related tags covering different \
angles (an emotion word, a situational word, a sound word if relevant), the same way \
exclude_tags already covers multiple synonyms. A single narrow tag risks zero results \
if that exact string isn't in this particular track's list, even when a close synonym \
would have matched fine.
- "Concise" still means no irrelevant padding -- it does not mean using the fewest \
possible tags for the concept you ARE targeting.

Examples:
- "victory parade music" -> tags=["victory", "triumphant", "celebration", "anthemic", "uplifting"]
- "songs for a rainy day" -> tags=["rainy", "rain", "melancholic", "wistful", "cozy", "reflective"]
- "something for studying" -> tags=["studying", "focus", "calm", "instrumental"]

## IMPLICIT EXCLUSIONS -- POSITIVE PHRASING CAN STILL MEAN "NOT X"

This applies broadly, not to any one axis: whenever a request implies "not X" -- even
phrased as a positive preference for the opposite of X -- that needs exclude_tags or
exclude_genres, not just an addition to the inclusive tags/genres list. tags and genres
are OR-matched (a track only needs ONE match to qualify), so adding the opposite quality
does NOT guarantee X gets removed -- a track can match on something else entirely (mood,
situation) and still pass through with the unwanted quality intact. Only exclude_tags/
exclude_genres actually removes matching tracks.

This shows up on any axis where "give me A" implies "not B" for a real, common category
in the data -- sound/production qualities, genre, lyrical content, etc. Ask: does this
request rule something OUT, even if worded as a preference for its opposite?

IMPORTANT for "not instrumental"/"with lyrics" specifically: the word "instrumental" is \
NOT reliably present as a tag even on tracks that clearly are instrumental -- solo piano, \
classical, and ambient tracks often use genre/sound words like "piano" instead. Rather than \
guessing at which genres tend to be instrumental, set require_lyrics=true -- this checks \
whether the track actually has identifiable lyrical content in the data (empty means \
instrumental or unknown), which works correctly regardless of genre.

Examples:
- "yoga music but with lyrics, not instrumental" -> tags=["yoga", "calm", "meditative"], require_lyrics=true
- "upbeat pop, not rock" -> genres=["pop"], exclude_genres=["rock"]
- "something acoustic, not electronic" -> tags=["acoustic"], exclude_tags=["synth", "electronic", "programmed"]

## GENRE

Only set genres when explicitly requested or clearly central to the request. Do not \
infer a genre from a mood stereotype -- "sad songs" does NOT imply indie/alternative rock.

## LISTENING HISTORY

Only use min_play_count/max_play_count/min_days_since_last_played when the user refers \
to their own relationship with the songs ("my favorites", "songs I play a lot", "deep \
cuts", "haven't heard in a while"). Do not apply these for ordinary mood or activity requests.

## SORTING

- relevance: default for specific mood/activity requests
- play_count: when familiarity/favorites matter
- recent: when recently-played tracks are requested
- random: when the user explicitly wants variety

## EXCLUDE_TAGS -- USE MULTIPLE SYNONYMS, NOT JUST THE USER'S EXACT WORD

IMPORTANT: exclude_tags is for mood/sound/situational CONCEPTS, never for artist names or
specific song titles.
- "No Lana Del Rey" or "nothing by Drake" -> exclude_artists, not exclude_tags.
- "not I Wanna Be Your Slave" or "remove Blinding Lights" -> exclude_tracks, not
exclude_tags or exclude_artists -- the user named one specific SONG, not an artist or a
mood/sound concept. Putting a song title or artist name in exclude_tags silently does
nothing, since it will never match against mood/sound/situational tags.

Matching is substring-based, not semantic -- a track tagged "intense" or "hard-hitting" \
will NOT be caught by excluding only "aggressive", even though it means the same thing. \
When the user asks to exclude a concept, provide 2-4 synonymous variants covering how \
that concept is likely actually tagged, not just their literal wording. For example:
- "not aggressive" -> exclude_tags=["aggressive", "intense", "harsh", "defiant"]
- "nothing too sad" -> exclude_tags=["sad", "melancholic", "heartbreak", "depressing"]
Check the real vocabulary sample in the tool description for likely actual phrasings.

IMPORTANT -- mood tags can be UNRELIABLE for intensity/aggression concepts specifically: \
this library's tagging sometimes favors flattering framing ("triumphant", "defiant", \
"confident") for genuinely intense-sounding tracks over harsher words, even when a track \
objectively has an aggressive sound. For "not aggressive" / "not harsh" / "not intense" \
style requests, ALSO include SONIC proxies in exclude_tags alongside mood synonyms, since \
sound characteristics tend to be tagged more consistently and objectively than subjective \
mood framing: "distorted guitar", "screaming", "harsh vocals", "heavy drums", "punchy drums". \
This applies specifically to aggression/intensity requests -- don't over-apply this pattern \
to unrelated exclusion concepts (e.g. excluding "sad" doesn't need sonic proxies).

## SIMILAR-TO REQUESTS

For "songs like X" or "similar to X by Y" requests, set similar_to_track (and
similar_to_artist if given) instead of trying to guess the track's attributes yourself
-- the real track will be looked up in the database and its actual genre/mood/energy/
sound used as the search basis. You can still set other fields to adjust the request,
e.g. "songs like Blinding Lights but calmer" -> similar_to_track="Blinding Lights",
energy_max="medium".

## DISCOVERY (NEW MUSIC)

Set discover_new=true ONLY when the user explicitly wants music they don't already \
have -- "recommend something new", "discover music like X", "I haven't heard this \
before", "surprise me". Do not set it for ordinary playlist or mood requests, even if \
the results happen to include unfamiliar tracks -- discover_new specifically triggers a \
search of new music OUTSIDE the user's library via Last.fm, seeded from either \
similar_to_track or the top library matches.

discovery_ratio follows FIXED TIERS based on how strongly "new" is expressed -- use the \
exact value for whichever tier matches, do not interpolate between them. Discovery is ON \
BY DEFAULT for ordinary requests -- only explicit "old"/"library only" language turns it off:

| User says... | discovery_ratio | discover_new |
|---|---|---|
| (ordinary mood/playlist request, no new/old language either way) | 0.3 (default) | true -- this is the default, not opt-in |
| "some new", "a few new ones", "mix in some new" | 0.6 | true |
| "new" / "something new" / "recommend new music" (mentions new, not qualified by "some" or "all") | 0.9 | true |
| "all new", "entirely new", "only new", "100% new" | 1.0 | true |
| "old", "all old", "from my library", "nothing new", "just my music" | 0.0 (discover_new=false, no discovery_ratio) | false |

Examples:
- "recommend something new like cardigan" -> mentions "new" plainly -> discovery_ratio=0.9
- "give me some new songs for a workout" -> "some new" -> discovery_ratio=0.6
- "a playlist that's entirely new" -> "entirely new" -> discovery_ratio=1.0
- "road trip playlist, all from my library" -> discover_new=false
- "high energy playlist" (no mention of new/old at all) -> discover_new=false (ordinary request, no discovery signal)

## OUTPUT

Only fill parameters relevant to the request -- don't invent constraints the user didn't \
imply. After results come back, give a short 2-3 sentence explanation of your \
interpretation. Do not list every track individually -- results are shown separately."""


def call_agent(client: Anthropic, user_prompt: str, search_tool: dict):
    """Returns (params, debug_json, similar_to_note, usage, discover_new, resolved_seed).
    usage is {'input_tokens', 'cache_write_tokens', 'cache_read_tokens', 'output_tokens', 'cost_usd'}.
    resolved_seed is the full seed track dict (with real artist_name) if a
    similar_to_track request was resolved, else None."""
    # Prompt caching: the tool definition (which embeds ~150 vocabulary tags)
    # and the system prompt are both large and mostly static across calls --
    # marking them cacheable means repeat queries within the 5-minute cache
    # window only pay full price for the small, genuinely new part of each
    # request (the user's actual message), not the ~3000+ tokens of fixed
    # schema/vocabulary resent every time. Order matters: tools, then system,
    # then messages, per Anthropic's caching rules.
    cached_tool = {**search_tool, "cache_control": {"type": "ephemeral"}}
    system_blocks = [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]

    response = client.messages.create(
        model=MODEL,
        max_tokens=600,
        system=system_blocks,
        tools=[cached_tool],
        tool_choice={"type": "tool", "name": "search_tracks"},
        messages=[{"role": "user", "content": user_prompt}],
    )

    tool_use = next(b for b in response.content if b.type == "tool_use")
    tool_input = tool_use.input

    # Cost tracking, accounting for prompt-cache pricing tiers (Haiku 4.5
    # standard API): $1.00/MTok fresh input, $1.25/MTok cache write (25%
    # premium), $0.10/MTok cache read (90% off), $5.00/MTok output.
    input_tokens = response.usage.input_tokens
    cache_write_tokens = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
    cache_read_tokens = getattr(response.usage, "cache_read_input_tokens", 0) or 0
    output_tokens = response.usage.output_tokens

    cost_usd = (
        input_tokens / 1_000_000 * 1.00
        + cache_write_tokens / 1_000_000 * 1.25
        + cache_read_tokens / 1_000_000 * 0.10
        + output_tokens / 1_000_000 * 5.00
    )
    usage = {
        "input_tokens": input_tokens,
        "cache_write_tokens": cache_write_tokens,
        "cache_read_tokens": cache_read_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
    }

    similar_to_note = None
    seed_params = {}
    resolved_seed = None

    similar_to_track = tool_input.get("similar_to_track")
    if similar_to_track:
        candidates = find_track(similar_to_track, tool_input.get("similar_to_artist"))
        if candidates:
            seed = candidates[0]
            seed_params = similar_track_params(seed)
            # Exclude ALL candidates sharing the seed's exact display name +
            # artist (not just the single chosen track_uri) -- duplicate
            # database rows with identical display names (different
            # releases/versions) would otherwise slip through as if they
            # were genuinely different, unrelated "similar" tracks.
            same_name_uris = [
                c["track_uri"] for c in candidates
                if c["track_name"].strip().lower() == seed["track_name"].strip().lower()
                and c["artist_name"].strip().lower() == seed["artist_name"].strip().lower()
            ]
            seed_params["exclude_uris"] = same_name_uris
            resolved_seed = seed
            similar_to_note = f"Using \"{seed['track_name']}\" - {seed['artist_name']} as the similarity seed."
            if len(candidates) > 1:
                similar_to_note += f" ({len(candidates)} possible matches found, picked your most-played version.)"
        else:
            similar_to_note = (
                f"Couldn't find \"{similar_to_track}\" in your library -- "
                f"falling back to a general interpretation of the request instead."
            )

    # Claude's own explicit values take priority over the seed track's
    # attributes -- but min/max pairs must come from the SAME source. If
    # Claude sets only energy_min (say) and the OTHER bound silently falls
    # back to the seed's derived value, the two bounds can end up inverted
    # (e.g. min='medium' from Claude, max='medium-low' from a low-energy
    # seed) -- the range-expansion code then silently swaps them to avoid
    # crashing, which collapses the intended range into something much
    # narrower/lower than intended. Fix: treat each (min, max) pair as a
    # unit -- if EITHER bound is explicitly set by Claude, both bounds come
    # from Claude (missing side left open/unbounded), never mixed with seed.
    def pick_pair(min_field: str, max_field: str):
        if tool_input.get(min_field) is not None or tool_input.get(max_field) is not None:
            return tool_input.get(min_field), tool_input.get(max_field)
        return seed_params.get(min_field), seed_params.get(max_field)

    energy_min, energy_max = pick_pair("energy_min", "energy_max")
    valence_min, valence_max = pick_pair("valence_min", "valence_max")
    pace_min, pace_max = pick_pair("pace_min", "pace_max")

    merged_tags = list(set((tool_input.get("tags") or []) + (seed_params.get("tags") or [])))

    # Resolve exclude_tracks (specific song titles) into real track_uris,
    # combined with whatever exclude_uris already came from similar_to_track's
    # own self-exclusion. Reuses the same "exclude ALL uris sharing this
    # display name" logic proven correct for the similar_to_track case --
    # if multiple database rows share the excluded title (different
    # releases/versions), all of them get excluded, not just one.
    exclude_track_uris = list(seed_params.get("exclude_uris", []))
    excluded_track_names = tool_input.get("exclude_tracks", [])
    for name in excluded_track_names:
        matches = find_track(name)
        exclude_track_uris.extend(m["track_uri"] for m in matches)
    exclude_track_uris = list(set(exclude_track_uris))

    params = SearchParams(
        tags=merged_tags,
        exclude_tags=tool_input.get("exclude_tags", []),
        genres=tool_input.get("genres") or seed_params.get("genres", []),
        exclude_artists=tool_input.get("exclude_artists", []),
        exclude_genres=tool_input.get("exclude_genres", []),
        require_lyrics=tool_input.get("require_lyrics", False),
        energy_min=energy_min,
        energy_max=energy_max,
        valence_min=valence_min,
        valence_max=valence_max,
        pace_min=pace_min,
        pace_max=pace_max,
        min_play_count=tool_input.get("min_play_count"),
        max_play_count=tool_input.get("max_play_count"),
        min_days_since_last_played=tool_input.get("min_days_since_last_played"),
        sort_by=tool_input.get("sort_by", "relevance"),
        limit=tool_input.get("limit", 25),
        exclude_uris=exclude_track_uris,
    )

    # Discovery is ON by default unless the request explicitly signals
    # library-only. Enforced here in code (not just via prompt wording)
    # for the same reason the ratio tiers are code-enforced: a silent
    # "discover_new absent -> treat as false" default meant ordinary
    # requests with no explicit new/old language got zero discovery,
    # which wasn't the intended behavior. Only an EXPLICIT discover_new:
    # false (the "old"/"library only" case) should suppress discovery now.
    discover_new = tool_input.get("discover_new", True) is not False
    discovery_ratio = tool_input.get("discovery_ratio")
    if discover_new and discovery_ratio is None:
        discovery_ratio = 0.3

    return params, json.dumps(tool_input, indent=2), similar_to_note, usage, discover_new, resolved_seed, discovery_ratio


def run_query(client: Anthropic, search_tool: dict, user_prompt: str, context: str = "") -> tuple[dict, list]:
    """Run one query -- used for both single-shot mode and each turn of an
    interactive session. Returns (tool_input_dict, playlist) so a calling
    loop can carry state into the next turn."""
    full_prompt = f"{context}\n\nNew instruction: {user_prompt}" if context else user_prompt

    print(f'Request: "{user_prompt}"\n')
    params, debug_params, similar_note, usage, discover_new, resolved_seed, discovery_ratio = call_agent(
        client, full_prompt, search_tool
    )
    if similar_note:
        print(similar_note + "\n")
    print("Parameters Claude chose:")
    print(debug_params)

    results = search_tracks(params)
    discovery_results = []
    tool_input_dict = json.loads(debug_params)

    if discover_new:
        print(f"\nSearching your library + discovering new music (ratio: {discovery_ratio:.0%} new)...")
        seeds = [(r["track_name"], r["artist_name"]) for r in results[:5]]
        if resolved_seed:
            seed_tuple = (resolved_seed["track_name"], resolved_seed["artist_name"])
            if seed_tuple not in seeds:
                seeds.insert(0, seed_tuple)

        if seeds:
            # Fetch enough discovery candidates to actually satisfy the
            # target ratio -- previously hardcoded to 20 regardless of how
            # many were needed, which silently capped high-ratio requests
            # (e.g. 0.9 or 1.0) well below their target: a 25-track
            # playlist at ratio=0.9 needs ~22 discovery tracks, but a flat
            # limit of 20 can never supply that many. +10 buffer accounts
            # for candidates lost to library/artist-exclusion filtering.
            needed = int(params.limit * discovery_ratio)
            fetch_limit = max(20, needed + 10)
            try:
                discovery_results = discover_new_music(
                    seeds, limit=fetch_limit, exclude_artists=params.exclude_artists,
                    exclude_tracks=tool_input_dict.get("exclude_tracks", []),
                )
            except EnvironmentError as e:
                print(f"Discovery unavailable: {e}")
    else:
        discovery_ratio = 0.0

    playlist = assemble_playlist(
        library_results=results,
        discovery_results=discovery_results,
        total_size=params.limit,
        discovery_ratio=discovery_ratio,
    )

    lib_count = sum(1 for t in playlist if t["source"] == "library")
    new_count = sum(1 for t in playlist if t["source"] == "new")
    print(f"\n--- Playlist ({len(playlist)} tracks: {lib_count} library, {new_count} new) ---")
    print(format_playlist(playlist))

    print(f"\n--- Cost ---")
    print(f"Fresh input: {usage['input_tokens']} | Cache write: {usage['cache_write_tokens']} | "
          f"Cache read: {usage['cache_read_tokens']} | Output: {usage['output_tokens']}")
    if usage['cache_read_tokens'] > 0:
        print(f"(Cache hit -- {usage['cache_read_tokens']} tokens read at 90% off)")
    elif usage['cache_write_tokens'] > 0:
        print(f"(Cache miss/first write -- next query within 5 min will be cheaper)")
    print(f"This query cost: ${usage['cost_usd']:.5f}")
    total_cost, total_queries = log_cost(usage['cost_usd'])
    print(f"Session total so far: ${total_cost:.4f} across {total_queries} quer{'y' if total_queries == 1 else 'ies'}")

    return json.loads(debug_params), playlist


def summarize_playlist(playlist: list[dict], max_tracks: int = 10) -> str:
    shown = playlist[:max_tracks]
    summary = ", ".join(f"{t['track_name']} - {t['artist_name']}" for t in shown)
    if len(playlist) > max_tracks:
        summary += f", and {len(playlist) - max_tracks} more"
    return summary


def print_help() -> None:
    print("""
Commands:
  <any request>   Make a playlist request, e.g. "high energy workout songs"
  push            Push the current playlist to Spotify (prompts for a name)
  new             Clear context -- next request starts fresh, unrelated to the last one
  help / ?        Show this message
  quit / exit / q Exit

Refining a previous playlist works naturally -- after any request, try things like:
  "make it more upbeat"
  "swap out the sad ones"
  "add more new music"
  "no Taylor Swift this time"
without repeating your whole original request -- Claude sees what you asked for
last time and builds on it.
""")


def push_to_spotify(playlist: list[dict]) -> None:
    if not playlist:
        print("No playlist to push yet -- make a request first.\n")
        return

    name = input("Playlist name: ").strip()
    if not name:
        print("Cancelled -- no name given.\n")
        return
    make_public = input("Make it public? (y/N): ").strip().lower() == "y"

    try:
        access_token = get_valid_access_token()
    except EnvironmentError as e:
        print(f"Can't push: {e}\n")
        return
    except Exception as e:
        print(f"Authorization failed: {e}\n")
        return

    print("Pushing to Spotify (resolving any new-music tracks first)...")
    try:
        result = push_playlist(access_token, playlist, name, public=make_public)
    except Exception as e:
        print(f"Push failed: {e}\n")
        return

    print(f"\nDone -- added {result['added_count']}/{len(playlist)} tracks.")
    if result["playlist_url"]:
        print(f"Open it here: {result['playlist_url']}")
    if result["skipped"]:
        print(f"\nCouldn't find a Spotify match for {len(result['skipped'])} track(s):")
        for t in result["skipped"]:
            print(f"  {t['track_name']} - {t['artist_name']}")
    print()


def interactive_loop(client: Anthropic, search_tool: dict) -> None:
    print("Interactive mode. Type a request, 'new' to clear context, 'help', or 'quit'.\n")
    last_tool_input = None
    last_playlist = None

    while True:
        try:
            user_input = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if user_input.lower() == "new":
            last_tool_input, last_playlist = None, None
            print("Context cleared -- next request starts fresh.\n")
            continue
        if user_input.lower() in ("help", "?"):
            print_help()
            continue
        if user_input.lower() == "push":
            push_to_spotify(last_playlist)
            continue

        context = ""
        if last_tool_input:
            context = f"Previous request parameters: {json.dumps(last_tool_input)}"
            if last_playlist:
                context += f"\nPrevious playlist included: {summarize_playlist(last_playlist)}"
            context += (
                "\n(If the new instruction below is a refinement of the previous request -- "
                "e.g. 'more upbeat', 'remove X', 'add more new music' -- build on the previous "
                "parameters above, keeping anything not explicitly changed. If it's an unrelated "
                "new request, ignore this context and start fresh.)"
            )

        try:
            last_tool_input, last_playlist = run_query(client, search_tool, user_input, context)
        except Exception as e:
            print(f"Error: {e}\n")
        print()


def main():
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not found in .env")

    client = Anthropic(api_key=api_key)

    print("Loading real tag vocabulary from your library...")
    vocabulary = get_tag_vocabulary()
    print(f"({len(vocabulary)} distinct tags found)\n")

    search_tool = build_search_tool(vocabulary)

    if len(sys.argv) >= 2:
        # Single-shot mode: python agent.py "prompt" -- unchanged behavior
        user_prompt = " ".join(sys.argv[1:])
        run_query(client, search_tool, user_prompt)
    else:
        # No argument: interactive mode
        interactive_loop(client, search_tool)


if __name__ == "__main__":
    main()