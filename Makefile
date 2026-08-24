.PHONY: setup test lint audit-wsd review-wsd audit-wsd-quality collect-tuning

setup:
	git submodule update --init --recursive
	python -m pip install -e third_party/PEEK
	python -m pip install -e '.[dev]'
	python -m pip install -e third_party/ultralytics
	python -m pip install -r third_party/yolov5/requirements.txt

test:
	pytest -q

lint:
	ruff check src tests

# Dataset source defaults to configs/dataset.yaml's wsd_root; override by
# exporting WSD_ROOT first, or by passing --source directly to the script.
audit-wsd:
	wsd-audit --output data/wsd.yaml

review-wsd:
	python scripts/review_wsd_duplicates.py --output docs/wsd-v71-duplicate-review.json

audit-wsd-quality:
	python scripts/audit_wsd_quality.py

collect-tuning:
	python scripts/collect_tuning_results.py
