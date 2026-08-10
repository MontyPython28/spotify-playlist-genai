import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import os

load_dotenv()

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
))

# Try fetching a known track (London Boy - your most played)
result = sp.track("spotify:track:1vrd6UOGamcKNGnSHJQlSt")
print(f"result is: {result}")
print(f"Duration: {result['duration_ms']}ms ({result['duration_ms']//1000}s)")
print(f"Popularity: {result['popularity']}")