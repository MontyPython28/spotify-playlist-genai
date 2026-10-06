import React from "react";
import ReactDOM from "react-dom/client";
import { MantineProvider, createTheme } from "@mantine/core";
import "@mantine/core/styles.css";
import "./styles.css";
import App from "./App.jsx";

// "Late-night record shop" theme: warm black-green surfaces, Spotify-ish
// green for the primary action, amber for the vinyl label / "new" marker.
const theme = createTheme({
  primaryColor: "groove",
  primaryShade: 5,
  colors: {
    groove: [
      "#e9fbef", "#c9f4d6", "#a2ecb9", "#76e39a", "#56dc82",
      "#4ad37a", "#3bb866", "#2e9653", "#217540", "#14542d",
    ],
    amber: [
      "#fff6e6", "#ffe8bf", "#ffd894", "#fdc768", "#f8b64a",
      "#f2a93a", "#d98f2a", "#b0721f", "#875615", "#5e3b0b",
    ],
  },
  fontFamily: "'Space Grotesk', sans-serif",
  fontFamilyMonospace: "'JetBrains Mono', monospace",
  headings: { fontFamily: "'Fraunces', serif", fontWeight: "400" },
  defaultRadius: "md",
});

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <MantineProvider theme={theme} defaultColorScheme="dark">
      <App />
    </MantineProvider>
  </React.StrictMode>
);