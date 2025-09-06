See repository PRD for the next phase (gain-map–first HDR pipeline).

Highlights:
- Reference Mode: extract/re-embed gain maps from phone HDR (HEIC/JPEG_R) or derive from HDR float vs SDR base.
- Predictive Mode: GMNet (pretrained) to predict gain map + Qmax from SDR only; apply guardrails.
- Exporters: libultrahdr (JPEG_R) primary; ImageIO (HEIC, JPEG_R on macOS 15+) secondary.
- Minimal API + upload UI; strict failure semantics; platform-aware tests.

Areas requiring research:
- GMNet repo/checkpoints/inference outputs.
- libultrahdr API-4 function signatures and build steps.
- ImageIO auxiliary DataDescription and ISO hdrgm CGImageMetadata specifics for macOS.

