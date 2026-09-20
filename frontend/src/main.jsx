import React from "react";
import ReactDOM from "react-dom/client";
import { MantineProvider, createTheme } from "@mantine/core";
import "@mantine/core/styles.css";
import App from "./App.jsx";

// Dark theme with a violet primary -- the one accent color from the design
// plan, used deliberately (the generate action, the "new" track marker)
// rather than sprinkled everywhere.
const theme = createTheme({
  primaryColor: "violet",
  fontFamily:
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
  headings: {
    // A serif for the one hero moment; sans everywhere else.
    fontFamily: "Georgia, 'Times New Roman', serif",
  },
});

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <MantineProvider theme={theme} defaultColorScheme="dark">
      <App />
    </MantineProvider>
  </React.StrictMode>
);