# Reviewed benchmark artifacts

These gzip-compressed JSON files are the exact reviewed runs behind
`docs/BENCHMARK_RESULTS.md`:

- `scifact-*` and `nfcorpus-*`: public BEIR baseline, candidate, and production-path runs;
- `hierarchical-1000-*`: generated baseline, candidate, and production-path runs;
- `hierarchical-100000-*` and `hierarchical-1000000-*`: standalone FTS engine
  scale implementation ablations (not full production-path runs).

Each JSON contains raw rankings, per-query metrics, aggregate metrics,
operational measurements, exact dependency/model metadata, executable source
tree hash, failures, and paired comparisons where a baseline was included.
`promotion-gate.json` reproduces the pooled 10,000-sample decision;
`source-manifest.json` maps the shared source-tree hash to each executable
file. `summary.csv` is a compact machine-readable view. Verify the published
files with:

```bash
shasum -a 256 -c benchmarks/results/checksums.sha256
```

Generated corpora and downloaded BEIR archives are intentionally not committed.
Their source URLs, checksums, and generator seed are in
`benchmarks/manifests/datasets.json`.

