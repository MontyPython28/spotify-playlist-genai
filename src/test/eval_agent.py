"""
eval_agent.py -- automated eval harness for the mood agent.

Runs a fixed set of test cases through the current agent, checks
automatable assertions per case, and reports a pass/fail score plus
manual-review items you should eyeball.

Use this before and after prompt changes to know whether you're
genuinely improving quality or just shifting where the failures land.

Run from the project root:
    python src/agent/eval_agent.py                # run all cases
    python src/agent/eval_agent.py --case 5       # run one case
    python src/agent/eval_agent.py --verbose      # show full playlists
"""

import argparse
import datetime
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# Make the agent module importable regardless of CWD
# This file lives in src/test/ -- the agent modules (agent.py,
# search_engine.py, discovery.py, playlist.py) live in the sibling
# src/agent/ folder, so add that to the import path explicitly.
sys.path.insert(0, str(Path(__file__).parent.parent / "agent"))

from dotenv import load_dotenv

from agent import (
    build_search_tool, call_agent, get_tag_vocabulary,
)
from search_engine import search_tracks
from discovery import discover_new_music
from playlist import assemble_playlist


@dataclass
class TestCase:
    """One eval case.

    checks: list of (name, fn) where fn takes (tool_input, playlist)
        and returns (passed: bool, detail: str). Detail is shown on fail.
    manual_review: optional prompt shown to the user for cases where the
        real quality judgment can't be automated -- flagged but not scored.
    """
    id: int
    category: str
    prompt: str
    checks: list[tuple[str, Callable]] = field(default_factory=list)
    manual_review: str = ""


# --- Reusable check helpers ------------------------------------------------

def check_param_equals(field: str, expected):
    def _check(tool_input, playlist):
        actual = tool_input.get(field)
        return (actual == expected, f"{field}={actual!r}, expected {expected!r}")
    return _check


def check_param_in(field: str, allowed: list):
    def _check(tool_input, playlist):
        actual = tool_input.get(field)
        return (actual in allowed, f"{field}={actual!r}, expected one of {allowed}")
    return _check


def check_param_set(field: str):
    """Field is present and non-empty."""
    def _check(tool_input, playlist):
        val = tool_input.get(field)
        return (bool(val), f"{field}={val!r} (expected non-empty)")
    return _check


def check_energy_at_least(level: str):
    ENERGY = ["very-low", "low", "medium-low", "medium", "medium-high", "high", "very-high"]
    def _check(tool_input, playlist):
        v = tool_input.get("energy_min")
        if v is None:
            return (False, "energy_min not set")
        return (ENERGY.index(v) >= ENERGY.index(level),
                f"energy_min={v}, expected >= {level}")
    return _check


def check_energy_at_most(level: str):
    ENERGY = ["very-low", "low", "medium-low", "medium", "medium-high", "high", "very-high"]
    def _check(tool_input, playlist):
        v = tool_input.get("energy_max")
        if v is None:
            return (False, "energy_max not set")
        return (ENERGY.index(v) <= ENERGY.index(level),
                f"energy_max={v}, expected <= {level}")
    return _check


def check_no_artist_in_playlist(artist_substring: str):
    def _check(tool_input, playlist):
        matches = [t["artist_name"] for t in playlist
                   if artist_substring.lower() in t["artist_name"].lower()]
        return (len(matches) == 0,
                f"found {len(matches)} matching tracks: {matches[:3]}")
    return _check


def check_tags_include_any(candidates: list[str]):
    def _check(tool_input, playlist):
        tags = [t.lower() for t in tool_input.get("tags", [])]
        hits = [c for c in candidates if any(c.lower() in t or t in c.lower() for t in tags)]
        return (bool(hits),
                f"tags={tags}, expected any of {candidates}, found: {hits}")
    return _check


def check_exclude_tags_include_any(candidates: list[str]):
    def _check(tool_input, playlist):
        tags = [t.lower() for t in tool_input.get("exclude_tags", [])]
        hits = [c for c in candidates if any(c.lower() in t or t in c.lower() for t in tags)]
        return (bool(hits),
                f"exclude_tags={tags}, expected any of {candidates}, found: {hits}")
    return _check


def check_playlist_min_size(n: int):
    def _check(tool_input, playlist):
        return (len(playlist) >= n, f"playlist size={len(playlist)}, expected >= {n}")
    return _check


def check_discovery_fraction(min_frac: float, max_frac: float):
    def _check(tool_input, playlist):
        if not playlist:
            return (False, "empty playlist")
        new_count = sum(1 for t in playlist if t["source"] == "new")
        frac = new_count / len(playlist)
        return (min_frac <= frac <= max_frac,
                f"discovery fraction={frac:.0%}, expected {min_frac:.0%}-{max_frac:.0%}")
    return _check


def check_no_track_in_playlist(track_substring: str):
    def _check(tool_input, playlist):
        matches = [t["track_name"] for t in playlist
                   if track_substring.lower() in t["track_name"].lower()]
        return (len(matches) == 0,
                f"found {len(matches)} matching tracks: {matches[:3]}")
    return _check


def check_param_includes(field: str, expected_value: str):
    def _check(tool_input, playlist):
        values = [v.lower() for v in tool_input.get(field, [])]
        return (expected_value.lower() in values, f"{field}={values}, expected to include {expected_value!r}")
    return _check


def check_no_crash():
    """Just verifies we got here at all -- for edge cases."""
    def _check(tool_input, playlist):
        return (True, "no crash")
    return _check


# --- The 15 test cases -----------------------------------------------------

TEST_CASES = [
    TestCase(1, "routing", "high energy workout playlist", [
        ("energy_min set to high or above", check_energy_at_least("high")),
        ("discover_new defaults to true (discovery is on by default now)", check_param_in("discover_new", [None, True])),
        ("playlist has tracks", check_playlist_min_size(10)),
    ]),

    TestCase(2, "routing", "recommend something new like cardigan", [
        ("similar_to_track set", check_param_set("similar_to_track")),
        ("discover_new=true", check_param_equals("discover_new", True)),
        ("discovery_ratio=0.9", check_param_equals("discovery_ratio", 0.9)),
        ("~90% new tracks", check_discovery_fraction(0.85, 1.0)),
    ]),

    TestCase(4, "routing", "road trip playlist, all from my library", [
        ("discover_new=false", check_param_in("discover_new", [None, False])),
        ("0% discovery in output", check_discovery_fraction(0.0, 0.0)),
    ]),

    TestCase(5, "routing", "songs like Blinding Lights but calmer", [
        ("similar_to_track set", check_param_set("similar_to_track")),
        ("energy_max set (override)", check_param_set("energy_max")),
        ("energy_max is medium or lower", check_energy_at_most("medium")),
    ]),

    TestCase(6, "exclusion", "high energy rock playlist but not I Wanna Be Your Slave by Maneskin", [
        ("track not in playlist", check_no_track_in_playlist("I Wanna Be Your Slave")),
    ]),

    TestCase(7, "exclusion", "high energy playlist but not aggressive", [
        ("energy_min set to at least medium-high", check_energy_at_least("medium-high")),
        ("exclude_tags includes aggression synonyms",
         check_exclude_tags_include_any(["aggressive", "harsh", "intense"])),
        ("exclude_tags includes sonic proxies",
         check_exclude_tags_include_any(["distorted", "screaming", "heavy drums", "punchy"])),
    ]),

    TestCase(8, "exclusion", "sad songs but nothing by Lana Del Rey", [
        ("exclude_artists includes Lana Del Rey",
         lambda ti, p: (
             any("lana" in a.lower() for a in ti.get("exclude_artists", [])),
             f"exclude_artists={ti.get('exclude_artists', [])}"
         )),
        ("no Lana Del Rey in playlist", check_no_artist_in_playlist("Lana Del Rey")),
    ]),

    TestCase(9, "nuance", "sad but energetic songs", [
        ("valence_max is negative or lower",
         lambda ti, p: (
             ti.get("valence_max") in ("negative", "very-negative", None) and
             ti.get("valence_min") in ("negative", "very-negative", None),
             f"valence_min={ti.get('valence_min')}, valence_max={ti.get('valence_max')}"
         )),
        ("energy_min at least medium-high (not low)",
         check_energy_at_least("medium-high")),
    ]),

    TestCase(10, "nuance", "happy but calm playlist", [
        ("valence_min positive or above",
         lambda ti, p: (
             ti.get("valence_min") in ("positive", "very-positive"),
             f"valence_min={ti.get('valence_min')}"
         )),
        ("energy_max low or medium-low", check_energy_at_most("medium-low")),
    ]),

    TestCase(12, "coverage", "songs for a rainy day", [
        ("tags include rainy-day concept or synonyms",
         check_tags_include_any(["rainy", "rain", "melancholic", "wistful", "reflective", "cozy"])),
        ("playlist returns at least 15 tracks", check_playlist_min_size(15)),
    ],
    manual_review="Do the tracks actually feel rainy-day appropriate?"),

    TestCase(13, "coverage", "victory parade music", [
        ("tags include triumphant/celebration concepts",
         check_tags_include_any(["triumphant", "celebration", "victory", "anthemic", "uplifting"])),
        ("energy_min at least medium-high", check_energy_at_least("medium-high")),
        ("playlist returns at least 15 tracks", check_playlist_min_size(15)),
    ]),

    TestCase(14, "edge", "songs like SomeFakeTrackThatDoesntExist12345", [
        ("no crash", check_no_crash()),
        ("playlist still produced (fallback path)", check_playlist_min_size(1)),
    ]),

    TestCase(15, "edge", "playlist entirely new music like cardigan", [
        ("discover_new=true", check_param_equals("discover_new", True)),
        ("discovery_ratio=1.0 exactly", check_param_equals("discovery_ratio", 1.0)),
        ("100% new tracks", check_discovery_fraction(1.0, 1.0)),
    ]),

    TestCase(16, "nuance", "calming yoga music but with lyrics, not instrumental", [
        ("require_lyrics is set to true",
         check_param_equals("require_lyrics", True)),
    ],
    manual_review="Do the returned tracks actually have vocals, not just avoid the word 'instrumental'?"),

    TestCase(17, "exclusion", "upbeat pop playlist, not rock", [
        ("exclude_genres includes rock", check_param_includes("exclude_genres", "rock")),
    ]),
]


# --- Runner ----------------------------------------------------------------

def _call_agent_with_retry(prompt: str, search_tool, max_retries: int = 5,
                           default_wait: float = 30.0):
    """Call the agent, retrying on TRANSIENT errors (rate-limit 429 and
    temporary overload 503).

    This lives ONLY in the eval harness -- the eval fires many requests in a
    burst and free-tier LLM providers (e.g. Gemini) rate-limit hard and also
    return transient 503 "high demand" errors, so the eval needs to wait and
    retry to complete. Live user queries deliberately do NOT retry (a user
    shouldn't wait for a playlist) -- the provider layer stays fast-fail.

    For 429s we honor the API's own suggested retry delay when present
    ("retry in 51.1s"); for 503s (which carry no delay) we use exponential
    backoff. Genuine (non-transient) errors are raised immediately -- no
    point retrying a real failure."""
    for attempt in range(max_retries):
        try:
            return call_agent(prompt, search_tool)
        except Exception as e:
            msg = str(e)
            # Retry on TRANSIENT errors: rate limits (429/quota) AND
            # temporary overload (503 UNAVAILABLE / "high demand"). Both are
            # "try again later" conditions, not genuine failures.
            is_rate_limit = (
                "429" in msg or "RESOURCE_EXHAUSTED" in msg
                or "rate limit" in msg.lower() or "quota" in msg.lower()
            )
            is_overloaded = (
                "503" in msg or "UNAVAILABLE" in msg
                or "high demand" in msg.lower() or "overloaded" in msg.lower()
            )
            is_transient = is_rate_limit or is_overloaded
            if not is_transient or attempt == max_retries - 1:
                raise

            # 429s often carry a specific retry delay ("retry in 47.4s") --
            # honor it. 503s don't, so use exponential backoff (5s, 10s, 20s,
            # ...) capped, which spaces out attempts without over-waiting.
            wait = None
            m = re.search(r"retry in (\d+(?:\.\d+)?)s", msg) or re.search(r"retryDelay['\"]?:\s*['\"]?(\d+(?:\.\d+)?)s", msg)
            if m:
                wait = float(m.group(1)) + 1.0  # small buffer past the stated delay
            if wait is None:
                wait = min(default_wait, 5.0 * (2 ** attempt))  # 5,10,20,30(cap)...
            reason = "rate-limited" if is_rate_limit else "model overloaded"
            print(f"       ({reason}; waiting {wait:.0f}s then retrying, "
                  f"attempt {attempt + 2}/{max_retries})")
            time.sleep(wait)


def run_single_case(
    tc: TestCase, search_tool: dict, verbose: bool
) -> dict:
    """Execute one test case end-to-end and return per-check results."""
    try:
        params, debug_str, _, usage, discover_new, resolved_seed, discovery_ratio = _call_agent_with_retry(
            tc.prompt, search_tool
        )
        tool_input = json.loads(debug_str)

        library_results = search_tracks(params)
        discovery_results = []
        if discover_new:
            seeds = [(r["track_name"], r["artist_name"]) for r in library_results[:5]]
            if resolved_seed:
                st = (resolved_seed["track_name"], resolved_seed["artist_name"])
                if st not in seeds:
                    seeds.insert(0, st)
            if seeds:
                # Same fetch-limit fix as agent.py's run_query() -- see
                # comment there for full rationale.
                needed = int(params.limit * discovery_ratio)
                fetch_limit = max(20, needed + 10)
                try:
                    discovery_results = discover_new_music(
                        seeds, limit=fetch_limit, exclude_artists=params.exclude_artists,
                        exclude_tracks=tool_input.get("exclude_tracks", []),
                    )
                except Exception:
                    pass
        else:
            discovery_ratio = 0.0

        playlist = assemble_playlist(
            library_results=library_results,
            discovery_results=discovery_results,
            total_size=params.limit,
            discovery_ratio=discovery_ratio,
        )

    except Exception as e:
        return {
            "id": tc.id, "category": tc.category, "prompt": tc.prompt,
            "error": str(e), "checks": [], "cost": 0,
        }

    check_results = []
    for name, fn in tc.checks:
        try:
            passed, detail = fn(tool_input, playlist)
        except Exception as e:
            passed, detail = False, f"check raised: {e}"
        check_results.append({"name": name, "passed": passed, "detail": detail})

    return {
        "id": tc.id, "category": tc.category, "prompt": tc.prompt,
        "tool_input": tool_input, "playlist": playlist,
        "checks": check_results, "cost": usage.get("cost_usd", 0),
        "manual_review": tc.manual_review,
    }


def print_result(result: dict, verbose: bool, max_tracks: int | None = 10) -> None:
    print(f"\n[{result['id']:>2}] {result['category']:<10} \"{result['prompt']}\"")

    if "error" in result:
        print(f"     ERROR: {result['error']}")
        return

    passed = sum(1 for c in result["checks"] if c["passed"])
    total = len(result["checks"])
    status = "PASS" if passed == total else "FAIL"
    print(f"     {status}  ({passed}/{total} checks, ${result['cost']:.5f})")

    for c in result["checks"]:
        marker = " OK " if c["passed"] else "FAIL"
        print(f"       [{marker}] {c['name']}  --  {c['detail']}")

    if result.get("manual_review"):
        print(f"     MANUAL REVIEW: {result['manual_review']}")

    if verbose and result.get("playlist"):
        print(f"     Playlist ({len(result['playlist'])} tracks):")
        shown = result["playlist"] if max_tracks is None else result["playlist"][:max_tracks]
        for t in shown:
            src = "[new]" if t["source"] == "new" else "     "
            print(f"       {src} {t['track_name']} - {t['artist_name']}")
        if max_tracks is not None and len(result["playlist"]) > max_tracks:
            print(f"       ... and {len(result['playlist']) - max_tracks} more")


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_checks = sum(len(r["checks"]) for r in results if "error" not in r)
    passed_checks = sum(
        sum(1 for c in r["checks"] if c["passed"])
        for r in results if "error" not in r
    )
    errored = sum(1 for r in results if "error" in r)
    fully_passing = sum(
        1 for r in results
        if "error" not in r and all(c["passed"] for c in r["checks"])
    )
    total_cost = sum(r.get("cost", 0) for r in results)

    print(f"Cases fully passing: {fully_passing}/{len(results)}")
    print(f"Individual checks:   {passed_checks}/{total_checks}")
    if errored:
        print(f"Errored cases:       {errored}")
    print(f"Total cost:          ${total_cost:.4f}")

    # By category
    print("\nBy category:")
    cats: dict[str, list[dict]] = {}
    for r in results:
        cats.setdefault(r["category"], []).append(r)
    for cat, rs in sorted(cats.items()):
        cat_passing = sum(
            1 for r in rs
            if "error" not in r and all(c["passed"] for c in r["checks"])
        )
        print(f"  {cat:<12} {cat_passing}/{len(rs)} fully passing")

    manual = [r for r in results if r.get("manual_review")]
    if manual:
        print("\nManual review items:")
        for r in manual:
            print(f"  [{r['id']}] {r['prompt']}: {r['manual_review']}")


def write_results_to_file(results: list[dict], output_dir: Path = Path("eval_results")) -> Path:
    """Write full results (including each case's actual playlist) to a
    timestamped JSON file, plus a human-readable .txt mirror of the
    console report. Returns the JSON file path.

    Keeping timestamped files (not overwriting a single output.json) means
    you can diff before/after a prompt change instead of only ever seeing
    the latest run.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = output_dir / f"eval_{timestamp}.json"
    txt_path = output_dir / f"eval_{timestamp}.txt"

    # JSON: full structured data, safe to diff programmatically later
    serializable = []
    for r in results:
        entry = {
            "id": r["id"], "category": r["category"], "prompt": r["prompt"],
            "cost": r.get("cost", 0),
        }
        if "error" in r:
            entry["error"] = r["error"]
        else:
            entry["tool_input"] = r.get("tool_input", {})
            entry["checks"] = r.get("checks", [])
            entry["manual_review"] = r.get("manual_review", "")
            entry["playlist"] = r.get("playlist", [])
        serializable.append(entry)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, default=str)

    # TXT: same format as the console report, for quick human reading
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        for r in results:
            print_result(r, verbose=True, max_tracks=None)  # full listing in the file
        if len(results) > 1:
            print_summary(results)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())

    return json_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, help="Run only this case ID")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show top of each playlist")
    args = parser.parse_args()

    load_dotenv()

    provider = os.getenv("LLM_PROVIDER", "anthropic")
    print(f"Setting up eval run (LLM_PROVIDER={provider})...")
    vocabulary = get_tag_vocabulary()
    search_tool = build_search_tool(vocabulary)

    cases = [tc for tc in TEST_CASES if args.case is None or tc.id == args.case]
    if not cases:
        print(f"No test case matches --case {args.case}")
        sys.exit(1)

    print(f"Running {len(cases)} case(s)...")
    results = []
    for tc in cases:
        result = run_single_case(tc, search_tool, args.verbose)
        results.append(result)
        print_result(result, args.verbose)

    if len(cases) > 1:
        print_summary(results)

    output_path = write_results_to_file(results)
    print(f"\nFull results written to: {output_path}")
    print(f"(and the matching .txt file alongside it)")


if __name__ == "__main__":
    main()