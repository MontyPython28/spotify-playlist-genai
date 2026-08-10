"""
playlist.py -- Phase 6c: blend library + discovery into one unified playlist.

Takes library search results and discovery results (two different shapes)
and produces a single interleaved playlist where discovery tracks are
woven throughout the library tracks at a configurable ratio, rather than
bolted on as a separate block at the end.

Each track in the output is labeled with its source ('library' or 'new')
so downstream consumers (display, Spotify push) can distinguish them.
"""

from __future__ import annotations


def assemble_playlist(
    library_results: list[dict],
    discovery_results: list[dict],
    total_size: int = 25,
    discovery_ratio: float = 0.3,
) -> list[dict]:
    """Blend library and discovery results into one interleaved playlist.

    Args:
        library_results: tracks from search_tracks() (have 'track_name',
            'artist_name', 'genre', 'energy', 'track_uri', etc.)
        discovery_results: tracks from discover_new_music() (have 'name',
            'artist', 'best_match', 'seed_count', 'source')
        total_size: target playlist length
        discovery_ratio: fraction of the playlist that should be discovery
            tracks (0.0 = library only, 1.0 = discovery only, 0.3 = default)

    Returns:
        A single flat list of dicts, each with at least:
            'track_name', 'artist_name', 'source' ('library' | 'new'),
            'genre' (if available), 'energy' (if available),
            'track_uri' (if available, library tracks only)
    """
    # Calculate how many of each to include
    n_discovery = min(
        int(total_size * discovery_ratio),
        len(discovery_results),
    )
    n_library = min(
        total_size - n_discovery,
        len(library_results),
    )
    # If we couldn't fill the discovery quota, backfill with more library
    if n_library + n_discovery < total_size:
        n_library = min(total_size - n_discovery, len(library_results))

    # Normalize both sources to a common shape
    lib_tracks = []
    for r in library_results[:n_library]:
        lib_tracks.append({
            "track_name": r["track_name"],
            "artist_name": r["artist_name"],
            "genre": r.get("genre", ""),
            "energy": r.get("energy", ""),
            "track_uri": r.get("track_uri"),
            "source": "library",
        })

    disc_tracks = []
    for r in discovery_results[:n_discovery]:
        disc_tracks.append({
            "track_name": r["name"],
            "artist_name": r["artist"],
            "genre": "",  # discovery tracks don't have genre metadata
            "energy": "",
            "track_uri": None,  # not in library, no Spotify URI known yet
            "source": "new",
        })

    # Interleave: distribute discovery tracks evenly throughout the
    # library tracks rather than appending them as a block.
    if not disc_tracks:
        return lib_tracks
    if not lib_tracks:
        return disc_tracks

    playlist = []
    # Calculate how often to insert a discovery track.
    # e.g. 17 library + 8 discovery → insert one discovery track roughly
    # every 2-3 library tracks.
    interval = max(1, n_library // (n_discovery + 1))
    disc_idx = 0

    for i, track in enumerate(lib_tracks):
        playlist.append(track)
        # After every `interval` library tracks, insert a discovery track
        if (i + 1) % interval == 0 and disc_idx < len(disc_tracks):
            playlist.append(disc_tracks[disc_idx])
            disc_idx += 1

    # Append any remaining discovery tracks that didn't get interleaved
    # (can happen when library list is short relative to discovery quota)
    while disc_idx < len(disc_tracks):
        playlist.append(disc_tracks[disc_idx])
        disc_idx += 1

    return playlist


def format_playlist(playlist: list[dict], show_numbers: bool = True) -> str:
    """Format a blended playlist for display, with source labels."""
    lines = []
    for i, t in enumerate(playlist, 1):
        label = "[new] " if t["source"] == "new" else ""
        genre_info = f" ({t['genre']}, {t['energy']} energy)" if t["genre"] else ""
        prefix = f"{i:>2}. " if show_numbers else "  "
        lines.append(f"{prefix}{label}{t['track_name']} - {t['artist_name']}{genre_info}")
    return "\n".join(lines)