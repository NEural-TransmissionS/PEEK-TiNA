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

audit-wsd:
	wsd-audit --source "$${WSD_ROOT:?Set WSD_ROOT to the WSD dataset directory}" --output data/wsd.yaml

review-wsd:
	python scripts/review_wsd_duplicates.py --source "$${WSD_ROOT:?Set WSD_ROOT to the WSD dataset directory}" --output docs/wsd-v71-duplicate-review.json

audit-wsd-quality:
	python scripts/audit_wsd_quality.py --source "$${WSD_ROOT:?Set WSD_ROOT to the WSD dataset directory}"

collect-tuning:
	python scripts/collect_tuning_results.py
