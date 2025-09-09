PY=uv run

# Defaults
IMG?=./images/02_edited_sdr.jpeg
OUT?=./02_gmnet_result.jpg
MODEL?=gmnet
PORT?=8001

.PHONY: help setup web dev process test clean build-ultrahdr

help:
	@echo "🎯 HDR Photo Restorer - Quick Start"
	@echo ""
	@echo "Development:"
	@echo "  make setup      - Install dependencies (first time setup)"
	@echo "  make web        - Start web service (main interface)"
	@echo "  make dev        - Start web service with auto-reload"
	@echo ""
	@echo "CLI Usage:"
	@echo "  make process    - Process single image via CLI"
	@echo "  make test       - Run validation tests"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean      - Remove generated outputs"
	@echo ""
	@echo "👉 New developers: run 'make setup && make web'"

setup:
	@echo "🔧 Installing dependencies..."
	uv sync
	@echo "✅ Setup complete! Run 'make web' to start the service."

web:
	@echo "🚀 Starting HDR web service on http://localhost:$(PORT)"
	@echo "Press Ctrl+C to stop"
	$(PY) python -c "from hdr_web.app import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=$(PORT))"

dev:
	@echo "🔄 Starting HDR web service with auto-reload on http://localhost:$(PORT)"
	@echo "Press Ctrl+C to stop"
	$(PY) uvicorn hdr_web.app:app --host 0.0.0.0 --port $(PORT) --reload

process:
	@test -n "$(IMG)" -a -n "$(OUT)" || { echo "Usage: make process IMG=./images/in.jpg OUT=./out.jpg [MODEL=gmnet]"; exit 1; }
	$(PY) hdr-process --img "$(IMG)" --out "$(OUT)" --model "$(MODEL)" --verbose

test:
	@echo "🧪 Running HDR validation tests..."
	$(PY) python test_hdr_validation.py

clean:
	@echo "🧹 Cleaning up generated files..."
	rm -rf outputs outputs_test outputs_dbg out_uv bin temp_web
	find images -name "*_sdr_preview.jpg" -delete || true
	find images -name "*_hdr.tiff" -delete || true
	find images -name "*_gainmap.png" -delete || true
	find images -name "*_gainmap.jpg" -delete || true
	find . -name "*.jpg.tmp" -delete || true
	@echo "✅ Cleanup complete"
