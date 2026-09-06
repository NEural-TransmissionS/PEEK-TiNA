.PHONY: test lint masks coco-demo report

# Run from the repository root: `make -C seg_masks test`

test:
	pytest -q tests

lint:
	ruff check src scripts tests

# GrabCut pseudo-masks for the WSD train split (needs the WSD export).
IMAGES ?= ../../datasets/satellite_components-71/train/images
OUT    ?= artifacts/masks/train
masks:
	python scripts/generate_masks.py --images-dir $(IMAGES) --output-root $(OUT)

# Pipeline check on COCO128 (no WSD needed): needs `third_party/datasets/coco128`
# (ultralytics downloads it) — see README.
coco-demo:
	python scripts/generate_masks.py \
	  --images-dir ../third_party/datasets/coco128/images/train2017 \
	  --output-root artifacts/masks/coco128

report:
	python scripts/feasibility_report.py $(OUT)/mask-manifest.json
