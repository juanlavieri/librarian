# Librarian benchmarks

This non-packaged suite evaluates passage retrieval on BEIR SciFact,
BEIR NFCorpus, and deterministic hierarchical/versioned corpora.

Install the benchmark dependencies:

```bash
pip install -e ".[benchmark]"
```

Run the frozen local baseline matrix:

```bash
python -m benchmarks.run --dataset scifact
python -m benchmarks.run --dataset nfcorpus
python -m benchmarks.run --dataset hierarchical-1000
```

Run semantic and fusion candidates:

```bash
python -m benchmarks.run --dataset scifact \
  --systems current-hybrid,dense-minilm,fts5-bm25,rrf-bm25-minilm
```

Every run writes a timestamped JSON artifact under `benchmarks/runs/`. Raw
rankings and generated/downloaded corpora are ignored by git. Reviewed,
compressed final artifacts belong under `benchmarks/results/`.

The current-code baseline uses Librarian's actual `HashingEmbedder`,
`LocalVectorStore`, and weighted `Retriever` fusion over identical benchmark
passages. Parsing and summarization are deliberately excluded so each
retrieval system receives the same corpus text.

