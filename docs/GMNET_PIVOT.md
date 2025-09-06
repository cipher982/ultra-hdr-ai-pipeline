Overview

- Objective: Deliver a one‑shot pipeline and minimal web service that outputs HDR images
using the Ultra HDR gain‑map standard (ISO 21496‑1) with two modes:
    - Reference Mode: If an HDR reference (e.g., iPhone/Android original) is available,
extract or derive its gain map and re‑embed with the user’s edited SDR base.
    - Predictive Mode: If only SDR input is available, predict a gain map (via a small
model such as GMNet) and package it with the SDR base.
- Outputs: Ultra HDR JPEG (JPEG_R) as the default, optional HEIC; SDR preview for web;
optional HDR float for QA.
- Focus: Treat gain map as the primary product. Hide container and metadata complexity
inside exporters for portability and reliability.

Personas & Use Cases

- Creator: Has original HDR phone photos and SDR edits; wants to restore HDR “pop” in a
shareable file.
- Editor: Has only SDR images; wants safe HDR boost without color artifacts.
- Operator: Runs CLI/Make or batch service to generate HDR assets.

Goals

- Gain‑map‑first pipeline that works with or without a reference HDR source.
- Produce Ultra HDR JPEGs that “pop” in macOS Preview/Photos (15+) and Android 14+ apps,
and HEIC for Apple workflows.
- Minimal website: upload SDR (+ optional reference) → return SDR preview + HDR download.
Clear, bounded failures.

Non‑Goals (This Phase)

- Large‑scale serving, asynchronous queues, or public multi‑tenant UX.
- Video or temporal coherence.
- Complex model training pipelines beyond a small fine‑tune loop (optional later).

Key Concepts (Glossary)

- SDR Base: The 8‑bit image (sRGB or Display P3) everyone can see.
- Gain Map: 8‑bit gray image encoding per‑pixel brightness boost; decoded via log2 within
metadata bounds.
- Qmax/HDRCapacityMax: Maximum allowed headroom (stops/log2) to apply the gain map on
HDR displays.
- JPEG_R (Ultra HDR JPEG): SDR base + embedded gain map + XMP hdrgm metadata;
backward‑compatible.
- HEIC Gain‑Map: Same idea packaged in HEIF; preferred on Apple when JPEG_R write is
not available.

User Stories

- As a creator, when I provide my original phone HDR and my edited SDR, I get back a
shareable HDR JPEG that looks like the original HDR in highlights but reflects my edit
for SDR users.
- As a user with only SDR, I upload my image and get a tasteful HDR “pop” JPEG that works
across macOS/Android, plus an SDR preview to compare.
- As an operator, I want a single CLI and Make targets to process images and run tests
without manual steps.

Functional Requirements

- Reference Mode
    - Input: SDR image; optional HDR reference (HEIC/JPEG_R or HDR float).
    - If HEIC/JPEG_R: extract SDR base, gain map, and hdrgm metadata; re‑embed gain map
with the edited SDR base (or feather near masked edit regions).
    - If HDR float: compute linear luminance ratio vs SDR base to derive gain map;
quantize and package with hdrgm metadata.
    - Output: Ultra HDR JPEG_R; HEIC optional; SDR preview; optional HDR float for QA.
    - Output: Ultra HDR JPEG_R; HEIC optional; SDR preview; optional HDR float for QA.
- 
Predictive Mode
    - Input: SDR image only.
    - Generate gain map via a pretrained gain‑map model (e.g., GMNet) without retraining;
return a scalar Qmax (or compute a safe default).
    - Optional guardrails: edge‑aware smoothing, optional face/skin limiting, highlight
caps.
    - Output: Ultra HDR JPEG_R; HEIC optional; SDR preview.
- 
Exporters
    - Primary: libultrahdr for JPEG_R (cross‑platform reference encoder).
    - Secondary: Apple ImageIO path for HEIC (and JPEG_R on macOS 15+).
    - XMP hdrgm metadata present and correct: Version, BaseRenditionIsHDR, GainMapMin/Max
(log2), Gamma, OffsetSDR/HDR, HDRCapacityMin/Max.
- 
CLI & API
    - CLI: one command that accepts IMG (SDR), optional REF (HDR reference), Boost (maps
to Qmax), and outputs HDR container + SDR preview.
    - API (FastAPI): POST /process (multipart form), GET /result, with timeouts and clear
JSON errors.

Non‑Functional Requirements

- Performance: ≤ 2s per 1024×768 image on Apple Silicon for pipeline excluding large
model inference; overall ≤ 20s hard timeout for web/API requests.
- Reliability: Deterministic outputs; no “fake” HDR containers; explicit non‑zero exit on
export failure.
- Portability: libultrahdr for Linux/macOS; ImageIO used opportunistically on macOS.
- Security: File type/size validation; EXIF strip or allowlist; no long‑term storage.

Data Flow

- Reference Mode (HEIC/JPEG_R)
    - Ingest: SDR edited base + original HDR file.
    - Extract: From HDR file, read gain map and hdrgm metadata.
    - Align: Verify dimensions/aspect; adjust or feather for edited regions.
    - Package: Base = edited SDR; Aux = gain map (8‑bit gray); hdrgm values from
reference (or recomputed).
    - Export: JPEG_R (libultrahdr); HEIC optional (ImageIO).
    - Export: JPEG_R (libultrahdr); HEIC optional (ImageIO).
- 
Reference Mode (HDR Float)
    - Ingest: SDR edited base + HDR linear float (TIFF/EXR).
    - Compute: gain_log2 = clamp(log2((L_HDR+ε)/(L_SDR+ε)), range).
    - Quantize: to 8‑bit gain map; set hdrgm fields (Gamma=1, Offsets=0–1/64, Capacity
bounds).
    - Export: as above.
- 
Predictive Mode
    - Ingest: SDR edited base.
    - Infer: gain_map_8bit + Qmax via GMNet (pretrained).
    - Guard: bilateral/guided smoothing, optional face cap; set hdrgm (CapacityMax=Qmax).
    - Export: as above.

Interfaces

- CLI
    - process: hdr-process --img <path> [--ref <path>] [--boost <stops>] [--out <path>]
    - outputs: <stem>_sdr_preview.jpg, <stem>_gainmap.jpg or .heic, <stem>_meta.json,
optional <stem>_hdr.tiff.
    - outputs: <stem>_sdr_preview.jpg, <stem>_gainmap.jpg or .heic, <stem>_meta.json,
optional <stem>_hdr.tiff.
- 
API
    - POST /process: multipart (img, optional ref, boost); returns job ID or result JSON
with file paths.
    - GET /result: returns SDR preview and HDR container.

Deliverables

- Gain‑map extraction utilities (from HEIC/JPEG_R).
- Gain‑map computation (from HDR float vs SDR base).
- GMNet inference integration (no training required initially).
- Exporters: libultrahdr (JPEG_R) + ImageIO (HEIC; JPEG_R on macOS 15+).
- CLI + FastAPI service + minimal upload UI.
- Tests: unit (math), integration (export/metadata), end‑to‑end (API with TestClient).

Research Tasks (Explicit)

- GMNet Research
    - Locate the GMNet repository/paper with pretrained checkpoints.
    - Understand its inference interface and outputs (gain map PNG, Qmax scalar, any
decode formulas).
    - Confirm expected GainMapMin/Max scaling and how to map to hdrgm.
    - Confirm expected GainMapMin/Max scaling and how to map to hdrgm.
- 
libultrahdr Integration
    - Build instructions and C/C++ binding strategy (API‑4: SDR JPEG + GM JPEG + metadata
→ JPEG_R).
    - Exact function names and parameter mapping (GainMapMin/Max log2, Gamma, Offsets,
HDRCapacity*).
    - Simple wrapper for Python/CLI.
- 
ImageIO Export Details (macOS)
    - For HEIC and JPEG_R (macOS 15+): exact DataDescription keys for aux image; confirm
kCGImageAuxiliaryDataTypeISOGainMap vs legacy HDRGainMap.
    - CGImageMetadata API for writing ISO hdrgm XMP; validation steps.
- 
Validation Tools
    - exiftool usage to verify MPF and XMP hdrgm tags; how to extract embedded gain map
for inspection.
    - Platform test matrices (Preview/Photos, Android Gallery).
- 
Color Management
    - Best practice for SDR base ICC (Display P3 vs sRGB) across platforms; guidance for
Instagram/web ingestion.

Acceptance Criteria

- Exporter emits Ultra HDR JPEG that renders as HDR in macOS Preview/Photos and Android
14+ Gallery.
- Reference Mode: Given a phone HDR HEIC/JPEG_R + edited SDR, outputs a shareable HDR
JPEG_R that matches the reference’s “pop” in highlights while preserving SDR edits.
- Predictive Mode: Given only SDR, outputs an HDR JPEG_R with controlled enhancement;
user can adjust Boost (Qmax).
- SDR preview looks natural and matches SDR base orientation/dimensions exactly; no
stretching.
- Tests verify:
    - hdrgm tags present with correct values.
    - MPF presence for JPEG_R; aux linkage for HEIC.
    - Luminance sanity: boosted highlights relative to SDR within safe bounds.

Testing Plan

- Unit
    - Gain map math: log2 ratios, quantization, offsets, gamma behavior.
    - Guardrails: bilateral smoothing radius/σ; optional face limiter.
    - Guardrails: bilateral smoothing radius/σ; optional face limiter.
- 
Integration
    - Extractor: load HEIC/JPEG_R, identify aux gain map presence, round‑trip write.
    - Exporter: file contains hdrgm XMP; MPF entries; extractable gain map.
- 
End‑to‑End
    - Reference Mode: HEIC + edited SDR → JPEG_R; verify pop visually and via metadata
detectors.
    - Predictive Mode: SDR → GMNet → JPEG_R; verify metadata and luminance metrics.
- 
Platform Awareness
    - macOS: run ImageIO HEIC export tests; JPEG_R where supported.
    - Linux: run libultrahdr path; skip ImageIO tests.

Security & Privacy

- Validate image type and dimensions; enforce max size/side length; reject unsupported
formats.
- Strip EXIF except allowlisted fields; embed ICC profile for base image.
- Remove uploads after processing; store outputs locally only for testing.

Performance

- Target ≤ 2s per 1024×768 for pipeline excluding heavyweight inference.
- Exporter and preprocessing operations must complete within a 20s request timeout in
service mode.

Milestones

- M1: libultrahdr path; CLI end‑to‑end (Reference Mode and Predictive Mode with
placeholder GM).
- M2: ImageIO HEIC export parity on macOS; JPEG_R on 15+.
- M3: GMNet integration (pretrained); Predictive Mode default; guardrails.
- M4: FastAPI service + minimal upload UI; platform‑aware tests.
- M5: Optional GMNet fine‑tune hooks and documentation.

Open Questions

- Preferred default ICC: sRGB or Display P3 for the SDR base (cross‑platform ingestion
differences)?
- Defaults for GainMapMin/Max if GMNet does not specify training range (start [-6, +6]
stops)?
- Offsets default: 0.0 or 1/64 — which is safer across devices?
- How aggressive should Boost (Qmax) slider be by default; scene‑based p95 vs fixed 3.0
stops?

Risks & Mitigations

- Exporter incompatibility: Use libultrahdr as primary; ImageIO as secondary where
available.
- Model artifacts: Start with pretrained GMNet; add guardrails; fine‑tune only if
necessary.
- Inconsistent HDR rendering: Validate via exiftool and across multiple viewers/devices;
adjust hdrgm fields accordingly.

Operational Notes

- Make Targets: setup, process (IMG/REF/Boost), test, clean.
- Logging: concise progress logs; explicit errors for exporter or model failures.
- No long‑running servers in tests; timeouts enforced on subprocesses.

This PRD defines the next phase: a gain‑map–first HDR pipeline shipping Ultra HDR
JPEG_R (and HEIC) with a clear Reference Mode and Predictive Mode, a robust exporter
layer, and a minimal website/API. Areas marked “Research Tasks” require targeted online
investigation (GMNet repo/checkpoints, libultrahdr API‑4, ImageIO details) before
implementation.