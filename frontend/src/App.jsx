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
  Alert,
  Modal,
  Switch,
  Anchor,
  Image,
} from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";

const SUGGESTIONS = [
  "sad songs but not too heavy",
  "something chill for studying",
  "like Blinding Lights, but calmer",
  "victory parade music",
];

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

  async function runQuery(text = prompt) {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setPushResult(null);
    try {
      const resp = await fetch("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: text }),
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
    <Box mih="100vh">
      <Container size="sm" py={64}>
        <Stack gap={40}>
          {/* Brand */}
          <Group justify="space-between">
            <Text ff="Fraunces, serif" fs="italic" size="lg">
              Mood Agent
            </Text>
            <Text ff="monospace" size="xs" c="amber.5" tt="uppercase" style={{ letterSpacing: "0.3em" }}>
              Side A
            </Text>
          </Group>

          {/* Hero: vinyl + headline */}
          <Stack gap="lg" align="center">
            <div className={`vinyl ${loading ? "is-spinning" : ""}`}>
              <div className="vinyl-label" />
            </div>
            <Title order={1} ta="center" style={{ fontSize: "3.25rem", lineHeight: 1 }}>
              What do you <em style={{ color: "var(--green)" }}>feel like?</em>
            </Title>
            <Text c="dimmed" ta="center">
              Describe a vibe, an activity, a mood — we'll press a playlist from your library.
            </Text>
          </Stack>

          {/* Prompt */}
          <Stack gap="sm">
            <div className="prompt-bar">
              <TextInput
                flex={1}
                size="lg"
                variant="unstyled"
                px="md"
                placeholder="calm songs for a rainy morning…"
                value={prompt}
                onChange={(e) => setPrompt(e.currentTarget.value)}
                onKeyDown={(e) => e.key === "Enter" && runQuery()}
                disabled={loading}
              />
              <Button size="lg" radius="xl" color="groove" c="#0b100d" onClick={() => runQuery()} loading={loading}>
                Drop the needle
              </Button>
            </div>
            {!result && !loading && (
              <Group gap="xs" justify="center">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    className="chip"
                    onClick={() => {
                      setPrompt(s);
                      runQuery(s);
                    }}
                  >
                    “{s}”
                  </button>
                ))}
              </Group>
            )}
          </Stack>

          {error && (
            <Alert color="red" variant="light" title="The record skipped">
              {error}
            </Alert>
          )}

          {pushResult && (
            <Alert color="groove" variant="light" title="Saved to Spotify">
              Added {pushResult.added_count} tracks.{" "}
              {pushResult.playlist_url && (
                <Anchor href={pushResult.playlist_url} target="_blank" c="groove.4">
                  Open playlist ↗
                </Anchor>
              )}
              {pushResult.skipped?.length > 0 && (
                <Text size="sm" mt="xs" c="dimmed">
                  Couldn't find {pushResult.skipped.length} track(s) on Spotify.
                </Text>
              )}
            </Alert>
          )}

          {loading && (
            <Text ta="center" ff="Fraunces, serif" fs="italic" c="dimmed">
              Digging through the crates…
            </Text>
          )}

          {result && (
            <Stack gap="md">
              <Group justify="space-between" align="flex-end" pb="sm" style={{ borderBottom: "1px solid var(--line)" }}>
                <div>
                  <Title order={2} size="1.75rem">
                    Tracklist
                  </Title>
                  <Text c="dimmed" size="sm" ff="monospace">
                    {result.library_count} from library
                    {result.new_count > 0 && ` · ${result.new_count} new`}
                  </Text>
                </div>
                <Button
                  variant="outline"
                  color="amber"
                  radius="xl"
                  onClick={() => {
                    setPlaylistName(prompt.slice(0, 40));
                    openPush();
                  }}
                >
                  Save to Spotify
                </Button>
              </Group>

              {result.similar_note && (
                <Text size="sm" c="dimmed" ff="Fraunces, serif" fs="italic">
                  {result.similar_note}
                </Text>
              )}

              <div>
                {result.tracks.map((t, i) => (
                  <TrackRow key={t.id} track={t} index={i} />
                ))}
              </div>
            </Stack>
          )}
        </Stack>
      </Container>

      {/* Push-to-Spotify modal */}
      <Modal
        opened={pushOpened}
        onClose={closePush}
        title={<Text ff="Fraunces, serif" size="xl">Press it to Spotify</Text>}
        centered
        radius="lg"
        styles={{ content: { background: "var(--surface)" }, header: { background: "var(--surface)" } }}
      >
        <Stack>
          <TextInput
            label="Playlist name"
            value={playlistName}
            onChange={(e) => setPlaylistName(e.currentTarget.value)}
            data-autofocus
          />
          <Switch
            label="Make public"
            color="groove"
            checked={makePublic}
            onChange={(e) => setMakePublic(e.currentTarget.checked)}
          />
          {pushError && (
            <Alert color="red" variant="light" title="Push failed">
              {pushError}
            </Alert>
          )}
          <Group justify="flex-end">
            <Button variant="subtle" color="gray" onClick={closePush}>
              Cancel
            </Button>
            <Button color="groove" c="#0b100d" radius="xl" onClick={pushToSpotify} loading={pushing}>
              Save
            </Button>
          </Group>
        </Stack>
      </Modal>
    </Box>
  );
}

// One track row, styled like a record sleeve tracklist. New discoveries get
// an amber left rule plus a small badge.
function TrackRow({ track, index }) {
  const isNew = track.source === "new";

  return (
    <div className={`track ${isNew ? "is-new" : ""}`}>
      <span className="track-num">{String(index + 1).padStart(2, "0")}</span>
      <Image
        src={track.image_url}
        w={44}
        h={44}
        radius="sm"
        fallbackSrc="data:image/svg+xml;charset=utf-8,%3Csvg xmlns='http://www.w3.org/2000/svg' width='44' height='44'%3E%3Crect width='44' height='44' fill='%231a231e'/%3E%3C/svg%3E"
      />
      <Box style={{ minWidth: 0 }}>
        <Text truncate fw={500} size="sm">
          {track.track_name}
        </Text>
        <Text truncate size="xs" c="dimmed">
          {track.artist_name}
          {track.album ? ` · ${track.album}` : ""}
        </Text>
      </Box>
      <Group gap="xs" wrap="nowrap">
        {track.genre && (
          <Text size="xs" c="dimmed" ff="monospace" visibleFrom="sm">
            {track.genre}
          </Text>
        )}
        {isNew && (
          <Badge color="amber" variant="light" size="sm">
            new
          </Badge>
        )}
      </Group>
    </div>
  );
}