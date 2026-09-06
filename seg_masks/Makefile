.PHONY: test lint masks

# Run from the repository root: `make -C seg_masks test`

test:
	pytest -q tests

lint:
	ruff check src scripts tests

# Generate GrabCut pseudo-masks for the WSD train split (needs the WSD export).
# Override: make -C seg_masks masks IMAGES=../datasets/satellite_components-71/train/images
IMAGES ?= ../../datasets/satellite_components-71/train/images
OUT    ?= artifacts/masks/train
masks:
	python scripts/generate_masks.py --images-dir $(IMAGES) --output-root $(OUT)
