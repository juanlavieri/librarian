# Retrieval Benchmarking

Librarian does not currently claim superior retrieval quality, latency, cost,
or scalability relative to Pinecone, Weaviate, or enterprise retrieval stacks.
Such claims require a controlled, reproducible head-to-head evaluation.

This document defines the minimum evidence required before making comparative
performance claims.

The first completed local evaluation is published in
[BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md). It does not include hosted-service
comparisons or model-graded answer quality.

## Questions the benchmark must answer

1. Does Librarian retrieve the evidence needed to answer a query?
2. How highly does it rank that evidence?
3. Do answers remain grounded in the retrieved sources?
4. What latency, indexing time, storage, and monetary cost does it incur?
5. How do those results change with corpus size and document structure?

## Required metrics

### Retrieval quality

- **Recall@k:** fraction of judged relevant passages retrieved in the first
  `k` results, macro-averaged over queries. Report at `k = 1, 3, 5, 10, 20`.
- **MRR:** mean reciprocal rank of the first relevant result.
- **nDCG@k:** ranking quality when judgments have graded relevance.
- **Passage precision@k:** proportion of retrieved passages judged relevant.

### Answer quality

- **Answer-grounding accuracy:** fraction of material answer claims supported
  by the cited evidence.
- **Citation correctness:** fraction of citations that entail the associated
  claim.
- **Answer completeness:** fraction of required reference-answer facts covered.
- **Unsupported-claim rate:** material claims not supported by retrieved
  evidence.

Answer evaluation must be reported separately from retrieval evaluation. A
model can answer incorrectly with good evidence or answer correctly despite
poor retrieval.

### Operational performance

- **Query latency:** p50 and p95 end-to-end retrieval latency.
- **Indexing time:** wall-clock time for initial indexing and incremental
  updates.
- **Storage:** catalog and index bytes, including source-independent metadata.
- **Cost:** embedding, model, hosted-index, compute, and storage cost under a
  stated pricing date and workload.
- **Failure behavior:** result and latency when an optional semantic provider is
  unavailable.

## Representative corpus sizes

Run the same evaluation at multiple scales, subject to available data:

- **Small:** approximately 1,000 passages.
- **Medium:** approximately 100,000 passages.
- **Large:** approximately 1,000,000 passages.

If a system cannot run at a given scale, report that explicitly rather than
extrapolating. Include total documents, passages, bytes, average passage length,
file types, hierarchy depth, update rate, and language distribution.

## Representative query sets

The benchmark should include:

- exact filename, path, identifier, and rare-term lookup;
- paraphrased semantic questions;
- passage-level factual questions;
- questions requiring numbers, tables, dates, or quotations;
- hierarchy questions such as “what is in this folder or section?”;
- current-version versus archived-version questions;
- ambiguous follow-ups requiring conversational context;
- unanswerable questions; and
- permission-sensitive questions when access control is implemented.

Use human relevance judgments made without knowledge of which system returned a
result. Publish annotation guidelines and inter-annotator agreement.

## Comparison protocol

For every compared system:

1. Use the same source corpus, query set, relevance judgments, and hardware
   class where possible.
2. State the exact product and version, index configuration, embedding model,
   reranker, chunking strategy, distance metric, filters, and query parameters.
3. Separate default configurations from tuned configurations.
4. Give each system a documented, comparable tuning budget.
5. Warm caches consistently and report cold and warm latency separately.
6. Run enough repetitions to report sample count and confidence intervals.
7. Preserve raw ranked results, timing data, cost calculations, and evaluation
   scripts.

Comparisons against hosted services must include network latency and dated
pricing assumptions. Architectural differences should be described separately
from measured outcomes.

## Baselines and ablations

At minimum, compare:

- lexical-only retrieval;
- semantic-only retrieval;
- Librarian’s configured hybrid retrieval;
- the same retrieval without structural roll-ups;
- the same retrieval without contextual chunk headers; and
- each external system being named in a public comparison.

The current open-source implementation defaults to SQLite FTS5/BM25. Lightweight
token overlap, weighted semantic/lexical fusion, and reciprocal-rank fusion
remain separately identified configurations for compatibility and controlled
evaluation.

## Reproducibility checklist

A publishable result must include:

- [ ] immutable corpus manifest or generation script;
- [ ] query set and relevance labels;
- [ ] exact dependency and model versions;
- [ ] complete configuration for every system;
- [ ] hardware and network description;
- [ ] random seeds where applicable;
- [ ] raw ranked results;
- [ ] metric computation code;
- [ ] latency and indexing measurements;
- [ ] storage and cost calculations;
- [ ] confidence intervals or repeated-run variability; and
- [ ] known limitations and excluded failures.

The completed local benchmark supports only the claims recorded in
[BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md). Continue to describe Librarian as
a **local, evidence-oriented retrieval system** with hierarchy and
provenance—not as a proven replacement for enterprise vector infrastructure.
