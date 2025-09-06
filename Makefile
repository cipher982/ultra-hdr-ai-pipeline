PY=uv run

# Defaults
IMG?=./images/02_edited_sdr.jpeg
OUT?=./images/02_gainmap.jpg
WEIGHTS?=hdrcnn_tf2.weights.h5

.PHONY: help setup process test clean build-ultrahdr

help:
	@echo "Targets:"
	@echo "  setup           - Create venv and install deps (uv sync)"
	@echo "  process         - One-shot SDR->HDR gain-map export"
	@echo "  test            - Run tests"
	@echo "  clean           - Remove generated outputs"
	@echo "  build-ultrahdr  - (Optional) Build libultrahdr if vendored"

setup:
	uv sync

process:
	@test -n "$(IMG)" -a -n "$(OUT)" || { echo "Usage: make process IMG=./images/in.jpg OUT=./images/out_gainmap.jpg [REF=opt]"; exit 1; }
	$(PY) hdr-process --img "$(IMG)" --out "$(OUT)" --weights "$(WEIGHTS)" $(if $(REF),--ref "$(REF)")

test:
	$(PY) pytest -q

clean:
	rm -rf outputs outputs_test outputs_dbg out_uv bin
	find images -name "*_sdr_preview.jpg" -delete || true
	find images -name "*_hdr.tiff" -delete || true
	find images -name "*_gainmap.png" -delete || true
	find images -name "*_gainmap.jpg" -delete || true
	find images -name "*_gainmap.heic" -delete || true

build-ultrahdr:
	@echo "Optional: build libultrahdr (vendored under tools/libultrahdr)"
	@echo "Refer to tools/libultrahdr/README.md for build instructions."
