.PHONY: test lint

# Run from the repository root: `make -C topo_prune test`
# The detector-free core (filtrations, certificate, metrics, descriptors) runs
# anywhere with gudhi + scipy + numpy. Everything that needs the checkpoint /
# WSD export is blocked on a GPU host -- see REPO_MAP.md B2, DEVIATIONS.md D5.

test:
	pytest -q tests

lint:
	ruff check src tests
