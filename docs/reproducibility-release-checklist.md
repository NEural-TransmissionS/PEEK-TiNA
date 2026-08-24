# Reproducibility release checklist

- [ ] Confirm the WSD distribution license and remove machine-local paths.
- [ ] Record the final repository and all submodule commits.
- [ ] Archive split fingerprints, duplicate review, quality audit, and calibration manifest.
- [ ] Archive every effective model argument, seed, environment manifest, and best checkpoint.
- [ ] Complete tuning without reading test metrics; freeze winners before test evaluation.
- [ ] Repeat final conditions across declared seeds and report uncertainty.
- [ ] Generate all tables and figures from committed CSV/JSON records.
- [ ] Separate desktop proxy and physical Jetson measurements.
- [ ] Verify PEEK API against the final announced upstream revision.
- [ ] Replace manuscript TBDs only where a corresponding artifact exists.
- [ ] Tag the release and deposit immutable code/artifact archives.
