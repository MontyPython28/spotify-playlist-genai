import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
 
// During development you run TWO servers: this Vite dev server (the UI, with
// hot reload) and the FastAPI backend (the agent). The proxy below forwards
// API calls from the Vite server to FastAPI so the browser can call /query
// and /push without cross-origin issues.
//
// For production ("npm run build"), Vite emits static files into ../static/,
// which FastAPI then serves directly -- so the built app and API share one
// origin and no proxy is needed.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../static",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/query": "http://127.0.0.1:8000",
      "/push": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
    },
  },
});
 
