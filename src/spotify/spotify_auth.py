"""
spotify_auth.py -- Spotify OAuth (Authorization Code Flow with PKCE).

PKCE means no client_secret is needed in the token exchange, even though
your app has one -- this is the current Spotify-recommended flow (the
older implicit grant was fully removed in Nov 2025).

First run: prints an authorization URL, you log in via browser, then
paste back the URL you were redirected to (the page will look broken/
blank since nothing is actually listening on that port -- that's fine,
the URL still has what we need in it).

After that: the refresh token is stored locally and used automatically,
no browser step needed again unless you revoke access.

Run standalone to test the auth flow in isolation:
    python spotify_auth.py
"""

import base64
import hashlib
import json
import os
import secrets
import time
import urllib.parse
from pathlib import Path

import requests
from dotenv import load_dotenv

AUTH_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
REDIRECT_URI = "http://127.0.0.1:8888/callback"
SCOPES = "playlist-modify-public playlist-modify-private"

TOKEN_FILE = Path("data/spotify_token.json")


def _generate_pkce_pair() -> tuple[str, str]:
    """Returns (code_verifier, code_challenge)."""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).decode("utf-8")
    verifier = verifier.rstrip("=")  # unreserved chars only, per PKCE spec

    challenge_bytes = hashlib.sha256(verifier.encode("utf-8")).digest()
    challenge = base64.urlsafe_b64encode(challenge_bytes).decode("utf-8").rstrip("=")

    return verifier, challenge


def _load_token() -> dict | None:
    if TOKEN_FILE.exists():
        with open(TOKEN_FILE, encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_token(token_data: dict) -> None:
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TOKEN_FILE, "w", encoding="utf-8") as f:
        json.dump(token_data, f)


def _exchange_code_for_token(client_id: str, code: str, code_verifier: str) -> dict:
    resp = requests.post(TOKEN_URL, data={
        "client_id": client_id,
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "code_verifier": code_verifier,
    })
    resp.raise_for_status()
    return resp.json()


def _refresh_access_token(client_id: str, refresh_token: str) -> dict:
    resp = requests.post(TOKEN_URL, data={
        "client_id": client_id,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    })
    resp.raise_for_status()
    return resp.json()


def _run_interactive_authorization(client_id: str) -> dict:
    """First-time authorization: opens the flow in the browser (you copy
    the URL yourself), exchanges the resulting code for tokens."""
    verifier, challenge = _generate_pkce_pair()
    state = secrets.token_urlsafe(16)

    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "code_challenge_method": "S256",
        "code_challenge": challenge,
        "scope": SCOPES,
        "state": state,
    }
    auth_url = f"{AUTH_URL}?{urllib.parse.urlencode(params)}"

    print("\n--- Spotify authorization needed ---")
    print("1. Open this URL in your browser and log in / approve access:\n")
    print(f"   {auth_url}\n")
    print("2. After approving, your browser will redirect to a URL that")
    print("   looks broken/blank (nothing is actually listening there --")
    print("   that's expected). Copy the FULL URL from your browser's")
    print("   address bar and paste it below.\n")

    redirected_url = input("Paste the full redirected URL here: ").strip()
    parsed = urllib.parse.urlparse(redirected_url)
    query = urllib.parse.parse_qs(parsed.query)

    if "error" in query:
        raise RuntimeError(f"Spotify authorization denied or failed: {query['error'][0]}")
    if "code" not in query:
        raise RuntimeError("Couldn't find 'code' in the pasted URL -- check you copied the full URL.")
    if query.get("state", [None])[0] != state:
        raise RuntimeError("State mismatch -- possible tampering, aborting.")

    code = query["code"][0]
    token_data = _exchange_code_for_token(client_id, code, verifier)

    token_data["obtained_at"] = time.time()
    _save_token(token_data)
    print("\nAuthorization successful -- token saved for future runs.\n")
    return token_data


def get_valid_access_token() -> str:
    """Main entry point. Returns a valid access token, refreshing or
    running the full interactive flow as needed."""
    load_dotenv()
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    if not client_id:
        raise EnvironmentError("SPOTIFY_CLIENT_ID not found in .env")

    token_data = _load_token()

    if token_data is None:
        token_data = _run_interactive_authorization(client_id)
        return token_data["access_token"]

    obtained_at = token_data.get("obtained_at", 0)
    expires_in = token_data.get("expires_in", 3600)
    # Refresh a bit early (60s buffer) to avoid using a token that expires
    # mid-request.
    if time.time() < obtained_at + expires_in - 60:
        return token_data["access_token"]

    refresh_token = token_data.get("refresh_token")
    if not refresh_token:
        # No refresh token available (shouldn't normally happen) -- fall
        # back to full re-authorization.
        token_data = _run_interactive_authorization(client_id)
        return token_data["access_token"]

    new_token_data = _refresh_access_token(client_id, refresh_token)
    # Spotify's refresh response sometimes omits refresh_token if it
    # didn't change -- keep the old one in that case.
    if "refresh_token" not in new_token_data:
        new_token_data["refresh_token"] = refresh_token
    new_token_data["obtained_at"] = time.time()
    _save_token(new_token_data)
    return new_token_data["access_token"]


if __name__ == "__main__":
    token = get_valid_access_token()
    print(f"Got access token (first 12 chars): {token[:12]}...")