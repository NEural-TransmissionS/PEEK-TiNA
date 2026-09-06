.PHONY: test lint p1 e1 e2 report

# Run from the repository root: `make -C topo_prune test`
# The detector-free core runs anywhere with gudhi + scipy + numpy.
# The P1 experiments need the CPU detector stack (torch + third_party/ultralytics)
# -- see README "Reproduce the P1 experiments". COCO-pretrained yolo26n.pt stands
# in for the WSD checkpoint (REPO_MAP.md B2); pass WEIGHTS=... when it lands.

WEIGHTS ?= yolo26n.pt

test:
	pytest -q tests

lint:
	ruff check src tests experiments

e1:
	python experiments/p1_certificate/run_p1.py --weights $(WEIGHTS)

e2:
	python experiments/p1_certificate/synthetic_p15.py

report:
	python src/topo_prune/report.py

p1: e1 e2 report
