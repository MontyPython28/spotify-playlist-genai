import { useState } from "react";
import {
  Container,
  Title,
  Text,
  TextInput,
  Button,
  Group,
  Stack,
  Box,
  Badge,
  Loader,
  Alert,
  Modal,
  Switch,
  Anchor,
  Image,
} from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";

export default function App() {
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null); // { tracks, library_count, new_count, similar_note, cost_usd }

  // Push modal state
  const [pushOpened, { open: openPush, close: closePush }] = useDisclosure(false);
  const [playlistName, setPlaylistName] = useState("");
  const [makePublic, setMakePublic] = useState(false);
  const [pushing, setPushing] = useState(false);
  const [pushResult, setPushResult] = useState(null);
  const [pushError, setPushError] = useState(null);

  async function runQuery() {
    if (!prompt.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setPushResult(null);
    try {
      const resp = await fetch("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });
      if (!resp.ok) {
        const detail = await resp.json().catch(() => ({}));
        throw new Error(detail.detail || `Request failed (${resp.status})`);
      }
      setResult(await resp.json());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function pushToSpotify() {
    if (!playlistName.trim() || !result) return;
    setPushing(true);
    setPushError(null);
    try {
      const resp = await fetch("/push", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tracks: result.tracks,
          name: playlistName,
          public: makePublic,
        }),
      });
      if (!resp.ok) {
        const detail = await resp.json().catch(() => ({}));
        throw new Error(detail.detail || `Push failed (${resp.status})`);
      }
      setPushResult(await resp.json());
      closePush();
    } catch (e) {
      setPushError(e.message);
    } finally {
      setPushing(false);
    }
  }

  return (
    <Box style={{ minHeight: "100vh", background: "#12131A", color: "#EDEDF2" }}>
      <Container size="sm" py={80}>
        {/* Hero: the prompt is the centerpiece */}
        <Stack gap="xl">
          <Stack gap="xs" align="center">
            <Title
              order={1}
              style={{ fontSize: "2.75rem", fontWeight: 500, textAlign: "center" }}
            >
              What do you feel like?
            </Title>
            <Text c="#8A8AA3" size="sm">
              Describe a vibe, an activity, a mood — get a playlist from your library
            </Text>
          </Stack>

          <Group gap="sm" wrap="nowrap">
            <TextInput
              flex={1}
              size="lg"
              placeholder="victory parade music... / calm songs for a rainy morning..."
              value={prompt}
              onChange={(e) => setPrompt(e.currentTarget.value)}
              onKeyDown={(e) => e.key === "Enter" && runQuery()}
              disabled={loading}
            />
            <Button size="lg" onClick={runQuery} loading={loading}>
              Generate
            </Button>
          </Group>

          {error && (
            <Alert color="red" title="Something went wrong">
              {error}
            </Alert>
          )}

          {pushResult && (
            <Alert color="violet" title="Saved to Spotify">
              Added {pushResult.added_count} tracks.{" "}
              {pushResult.playlist_url && (
                <Anchor href={pushResult.playlist_url} target="_blank">
                  Open playlist
                </Anchor>
              )}
              {pushResult.skipped?.length > 0 && (
                <Text size="sm" mt="xs" c="#8A8AA3">
                  Couldn't find {pushResult.skipped.length} track(s) on Spotify.
                </Text>
              )}
            </Alert>
          )}

          {loading && (
            <Group justify="center" py="xl">
              <Loader color="violet" />
            </Group>
          )}

          {result && (
            <Stack gap="md">
              <Group justify="space-between" align="center">
                <Text c="#8A8AA3" size="sm">
                  {result.library_count} from library
                  {result.new_count > 0 && ` · ${result.new_count} new`}
                </Text>
                <Button
                  variant="light"
                  onClick={() => {
                    setPlaylistName(prompt.slice(0, 40));
                    openPush();
                  }}
                >
                  Save to Spotify
                </Button>
              </Group>

              {result.similar_note && (
                <Text size="sm" c="#8A8AA3" fs="italic">
                  {result.similar_note}
                </Text>
              )}

              <Stack gap={0}>
                {result.tracks.map((t) => (
                  <TrackRow key={t.id} track={t} />
                ))}
              </Stack>
            </Stack>
          )}
        </Stack>
      </Container>

      {/* Push-to-Spotify modal */}
      <Modal opened={pushOpened} onClose={closePush} title="Save to Spotify" centered>
        <Stack>
          <TextInput
            label="Playlist name"
            value={playlistName}
            onChange={(e) => setPlaylistName(e.currentTarget.value)}
            data-autofocus
          />
          <Switch
            label="Make public"
            checked={makePublic}
            onChange={(e) => setMakePublic(e.currentTarget.checked)}
          />
          {pushError && (
            <Alert color="red" title="Push failed">
              {pushError}
            </Alert>
          )}
          <Group justify="flex-end">
            <Button variant="default" onClick={closePush}>
              Cancel
            </Button>
            <Button onClick={pushToSpotify} loading={pushing}>
              Save
            </Button>
          </Group>
        </Stack>
      </Modal>
    </Box>
  );
}

// One track row. Library vs. new distinguished by a left-edge rule (violet
// for new), per the design plan -- not scattered badges.
function TrackRow({ track }) {
  const isNew = track.source === "new";
  const [hovered, setHovered] = useState(false);

  return (
    <Group
      wrap="nowrap"
      gap="md"
      py="xs"
      px="md"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        borderLeft: `2px solid ${isNew ? "#A66CFF" : "transparent"}`,
        background: hovered ? "#1C1D28" : "transparent",
        transition: "background 120ms ease",
        cursor: "default",
      }}
    >
      {/* Album art thumbnail (every track is Spotify-resolved, so this is
          always present; a neutral box shows if art is somehow missing) */}
      <Image
        src={track.image_url}
        w={44}
        h={44}
        radius="sm"
        fallbackSrc="data:image/svg+xml;charset=utf-8,%3Csvg xmlns='http://www.w3.org/2000/svg' width='44' height='44'%3E%3Crect width='44' height='44' fill='%232A2B38'/%3E%3C/svg%3E"
        style={{ flexShrink: 0 }}
      />

      <Box style={{ minWidth: 0, flex: 1 }}>
        <Text truncate fw={500} size="sm">
          {track.track_name}
        </Text>
        <Text truncate size="xs" c="#8A8AA3">
          {track.artist_name}
          {track.album ? ` \u00b7 ${track.album}` : ""}
        </Text>
      </Box>

      <Group gap="xs" wrap="nowrap" style={{ flexShrink: 0 }}>
        {track.genre && (
          <Text size="xs" c="#6C6C82" visibleFrom="sm">
            {track.genre}
          </Text>
        )}
        {isNew && (
          <Badge color="violet" variant="light" size="sm">
            new
          </Badge>
        )}
      </Group>
    </Group>
  );
}