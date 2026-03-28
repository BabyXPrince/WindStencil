# Figures and diagrams

| File | Purpose |
|------|---------|
| `repo-layout.svg` | High-level directory roles; embedded in `docs/ARCHITECTURE.md`. |

## Adding paper-quality figures

For the camera-ready or GitHub “paper parity” visuals (e.g. plane-scan schematic, roofline plots):

1. Export **SVG** or **PNG (≥600 dpi)** from your paper sources.
2. Place them here with stable names, e.g. `plane-scan.svg`, `roofline-mi200.svg`.
3. Link from **`README.md`** or **`docs/ARCHITECTURE.md`** with relative paths, e.g. `docs/images/plane-scan.svg`.

Avoid committing publisher-copyright PDFs; prefer originals you own or CC-licensed plots.
