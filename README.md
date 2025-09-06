Ultra HDR Gain‑Map Pipeline (MVP)
=================================

<svg xmlns="http://www.w3.org/2000/svg" width="512" height="256" viewBox="0 0 512 256" role="img" aria-labelledby="t d">
  <title id="t">HDR Gain-Map Logo</title><desc id="d">SDR plus gain map equals HDR</desc>
  <!-- double stroke for contrast on light/dark -->
  <rect x="6" y="6" width="500" height="244" rx="20" fill="none" stroke="white" stroke-width="6"/>
  <rect x="6" y="6" width="500" height="244" rx="20" fill="none" stroke="black" stroke-width="2"/>

  <!-- Left: SDR -->
  <text x="70" y="135" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto" font-weight="700" font-size="64" fill="#475569">SDR</text>

  <!-- Multiply sign -->
  <text x="190" y="135" font-family="system-ui" font-size="56" fill="#334155">×</text>

  <!-- Gain-map tile -->
  <g transform="translate(230,78)">
    <rect x="0" y="0" width="68" height="68" rx="8" fill="#e5e7eb" stroke="#0f172a" stroke-width="1.5"/>
    <!-- grayscale grid to suggest gain map -->
    <defs>
      <linearGradient id="gm" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#a3a3a3"/><stop offset="100%" stop-color="#f5f5f5"/>
      </linearGradient>
    </defs>
    <rect x="6" y="6" width="56" height="56" fill="url(#gm)" rx="4" stroke="#334155" stroke-width="0.8"/>
    <!-- 4x4 grid lines -->
    <g stroke="#64748b" stroke-width="0.7" opacity="0.8">
      <path d="M20 6 V62 M34 6 V62 M48 6 V62"/>
      <path d="M6 20 H62 M6 34 H62 M6 48 H62"/>
    </g>
    <text x="34" y="88" text-anchor="middle" font-family="system-ui" font-size="12" fill="#334155">gain&nbsp;map</text>
  </g>

  <!-- Equals -->
  <text x="320" y="135" font-family="system-ui" font-size="56" fill="#334155">=</text>

  <!-- Right: HDR with subtle "glow" -->
  <defs>
    <radialGradient id="glow" cx="75%" cy="40%" r="70%">
      <stop offset="0%" stop-color="#ffd166" stop-opacity="0.9"/>
      <stop offset="70%" stop-color="#fca311" stop-opacity="0.25"/>
      <stop offset="100%" stop-color="#f59e0b" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="hdrFill" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#ffb703"/><stop offset="100%" stop-color="#fb5607"/>
    </linearGradient>
  </defs>
  <circle cx="430" cy="108" r="62" fill="url(#glow)"/>
  <text x="395" y="135" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto" font-weight="900" font-size="64" fill="url(#hdrFill)" stroke="#111827" stroke-width="1.2">HDR</text>

  <!-- Subtext -->
  <text x="256" y="214" text-anchor="middle" font-family="system-ui" font-size="14" fill="#334155">SDR base × gain map = perceived HDR</text>
</svg>

This repository implements a gain‑map–first pipeline for producing HDR images
that render as HDR on macOS 15+/iOS 18+ (Preview/Photos), Android 14+ (Ultra HDR),
and social apps that support JPEG_R.

Outputs
- Ultra HDR JPEG (JPEG_R) — default; backward‑compatible.
- HEIC with gain map — optional on macOS.
- SDR preview (for web/UI and debugging).
- Optional HDR float TIFF (QA).

Modes
- Reference Mode (preferred): If you have a phone HDR original (HEIC/JPEG_R) or
  an HDR float reference, extract/derive the gain map and re‑embed it with your
  edited SDR base.
- Predictive Mode: If you only have SDR, predict a gain map (e.g., GMNet) and
  package it with the SDR base. A “Boost” control maps to HDRCapacityMax.

Exporters
- Primary: libultrahdr (JPEG_R; cross‑platform).
- Secondary: Apple ImageIO (HEIC; JPEG_R on macOS 15+).

Quick Start
- `make setup`
- `make process IMG=./images/02_edited_sdr.jpeg OUT=./images/02_gainmap.jpg`

Notes
- Place large HDR references under `images/` locally; do not commit them.
- `samples/` contains tiny images suitable for tests.
- The legacy SDR→HDR TF2 code has been archived; the next phase focuses on
  gain‑map–first export. See `docs/PRD_NEXT_PHASE.md`.
