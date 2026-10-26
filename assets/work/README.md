# Images and playable work

- `rollout-000.png` through `rollout-003.png` were extracted unchanged from page 7,
  figure 3, of the site's existing `assets/papers/world-model-rollout-audit.pdf`.
  They show the recorded frame and 1-, 8-, and 32-call predictions of the same
  8-second endpoint, in that order. The paper describes the clip-selection rule
  and the different display crop used for the recorded frame. The homepage
  retains that qualification and links to the figure. These are illustrative
  frames from one clip, not aggregate results.
- `mincut-cordon.png` is the original `docs/cordon.png` from Saket Atreya's
  [Min-Cut Muncher](https://github.com/saketatreya/mincut-pacman). It shows a
  deterministic game state with the ghosts holding a minimum vertex cut.
- The original `writings/rat__3.gif` is retained unchanged. The still and native
  pointer in `assets/` were derived from it in an earlier site revision.

The playable game at `games/mincut/index.html` is copied from the project's
self-contained `dist/index.html`, with its document title, return/source navigation, and a no-JavaScript notice
added. Source bundle SHA-256:
`27e22e220a5a1bbbd8835036b88a7370f1e0d5a0e6dbb1a33886fc7f7dd94ca2`.
Its embedded Press Start 2P font license is retained at
`games/mincut/FONT-LICENSE.txt`. The game engine, art, and controls are unmodified.

Illustrations and game assets are served locally. The site wordmark uses Archivo Black from Google Fonts.
