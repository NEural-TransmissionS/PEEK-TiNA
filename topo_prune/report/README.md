# P0--P1 progress report

`p1_report.tex` --- a deep writeup of the certificate design and the E1/E2
experiments, with a threats-to-validity section and an extended future-work
section (IBP worst-case bound, wider prune scope, the P2 two-sided intervention,
the Euler control, P4 restoration, on-device measurement).

Every number in Tables 2 and 3 is transcribed from the run manifests in
`../experiments/p1_certificate/runs/` (and matches `../docs/p1-results.md`, which
`report.py` generates). If you re-run the experiments, update the two tables.

## Build

Any LaTeX toolchain with `siunitx`, `booktabs`, `cleveref`, `hyperref`:

```bash
tectonic p1_report.tex          # self-contained, fetches packages on first run
# or
latexmk -pdf p1_report.tex
```

`p1_report.pdf` (14 pp.) is committed for readers without LaTeX.
