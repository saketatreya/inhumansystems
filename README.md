# INHUMAN SYSTEMS

> *Accelerating beyond the human.*

`INHUMAN SYSTEMS` is a minimalist, high-signal, zero-dependency web interface built for text publication and network visualization. It is designed around the architectural philosophy of "suckless" web design: maximum visual density and mathematical precision with minimal overhead.

## Architecture

- **Zero-Dependency Philosophy:** The platform relies exclusively on vanilla ES6 JavaScript and CSS. The only external resource is `marked.js` (loaded via CDN) for Markdown text rendering, preserving a brutalist, terminal-inspired frontend.
- **Dynamic Render Chain:** All `.md` files in `/writings` are parsed dynamically into HTML without a build step. Content indexing and the graph view are entirely driven by a lightweight `manifest.json`.
- **Generative Entropy / High-DPI Visuals:** Custom `<canvas>` elements run complex mathematical simulations directly in the browser:
  - The header features a daily-seeded **Lorenz Attractor**, generating chaotic differential trajectories that elegantly scale to the user's local date-hash.
  - Section dividers run real-time **Stephen Wolfram's Rule 30 Cellular Automata**, endlessly computing downward non-repeating patterns.
- **Topological Latent Space:** The embedded `/graph.html` visualizes semantic distances and conceptual linkages via a custom-implemented force-directed collision graph.

## Local Deployment

Due to modern browser CORS (Cross-Origin Resource Sharing) policies restricting `fetch()` loading over the `file://` protocol, the site must be served via a local HTTP server.

```bash
# Using Python 3's built-in server (from the local repository directory)
python3 -m http.server 8888
```

Then navigate your web browser to `http://localhost:8888`.

## Maintained by

**Saket Atreya**
Building at the intersection of machine intelligence, nonlinear dynamics, and inhuman theory.
