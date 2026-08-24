# Contributing

Keep experimental claims traceable to machine-readable artifacts. Do not alter WSD splits, inspect test results during selection, update vendored commits, or change frozen seeds/protocols without a separate documented experiment identifier. Run `make test` and `make lint`, preserve manifests and exact commands, and never commit private dataset images or machine-local dataset paths.

Changes to dataset provenance, model selection, pruning criteria, or evaluation protocol should update the README to distinguish exploratory from confirmatory results. Generated tables must come from committed CSV/JSON records rather than manual transcription.
