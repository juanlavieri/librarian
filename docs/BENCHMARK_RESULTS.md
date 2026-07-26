# Retrieval benchmark results

Run date: 2026-07-22 UTC  
Code baseline: `6d224c70b568f915a46ce989f4b42e1a0f80b4a9` plus the documented
benchmark harness changes  
Production-path source tree: recorded as `source_tree_sha256` in each
production artifact  
Runtime: Python 3.12.13, macOS 26.3.1 arm64  
Model: `sentence-transformers/all-MiniLM-L6-v2` (where shown)

## Decision

SQLite FTS5/BM25 is the promoted default retrieval strategy. It was the only
candidate that cleared the declared gate:

- it improved macro nDCG@10 and Recall@10 over the current weighted hybrid,
  lexical overlap, and hashing-cosine baselines;
- paired 95% bootstrap confidence intervals excluded zero;
- it did not regress an individual dataset;
- its warm p95 latency was lower than the current weighted hybrid on all
  measured corpora;
- forced semantic-provider failure produced the same rankings as BM25 rather
  than an empty or failed search.

The promotion numbers below come from `ProductionFTS5System`, which exercises
the shipped `LocalVectorStore`, `Retriever`, contextual records, roll-ups,
filters, candidate-pool handling, and deduplication over identical passages.
Chunk-deepening behavior is covered by focused regression tests but is not
measured by the passage-level BEIR runs.
The standalone FTS candidate remains in the artifacts as an implementation
ablation, not as the evidence for the production default.

RRF over BM25 and MiniLM led both public datasets, but was not promoted because
hierarchical nDCG@10 fell from 1.0000 to 0.9797, slightly beyond the declared
2% per-dataset regression limit.

## Quality results

| Dataset | System | nDCG@10 | Recall@10 | Warm p95 |
|---|---|---:|---:|---:|
| SciFact (5,183 docs / 300 queries) | Current weighted hybrid | 0.4151 | 0.5336 | 197.70 ms |
| SciFact | Production FTS5/BM25 | **0.6716** | **0.7986** | **12.46 ms** |
| SciFact | MiniLM dense | 0.6469 | 0.7883 | 11.73 ms |
| SciFact | BM25 + MiniLM RRF | 0.6863 | 0.8044 | 17.72 ms |
| NFCorpus (3,633 docs / 323 queries) | Current weighted hybrid | 0.2242 | 0.1092 | 148.79 ms |
| NFCorpus | Production FTS5/BM25 | **0.3033** | **0.1457** | **26.07 ms** |
| NFCorpus | MiniLM dense | 0.3162 | 0.1547 | 9.84 ms |
| NFCorpus | BM25 + MiniLM RRF | 0.3391 | 0.1702 | 8.22 ms |
| Hierarchical (1,000 passages / 200 queries) | Current weighted hybrid | **1.0000** | **1.0000** | 16.06 ms |
| Hierarchical | Production FTS5/BM25 | **1.0000** | **1.0000** | **2.41 ms** |
| Hierarchical | MiniLM dense | 0.9169 | 1.0000 | 4.31 ms |
| Hierarchical | BM25 + MiniLM RRF | 0.9797 | 1.0000 | 5.20 ms |

The original local ablations are retained in the raw artifacts. Notably,
token-overlap lexical retrieval beat the existing weighted hybrid on SciFact
(0.4606 versus 0.4151 nDCG@10), while the weighted hybrid beat token overlap
on NFCorpus (0.2242 versus 0.1985). This is why no universal claim is made for
the old fusion weights.

## Aggregate paired confidence intervals

The following query-paired bootstrap deltas are FTS5/BM25 minus each baseline,
pooled across SciFact, NFCorpus, and the 1K hierarchical corpus (823 queries,
10,000 bootstrap samples).

| Baseline | Metric | Mean delta | 95% CI |
|---|---|---:|---:|
| Current weighted hybrid | nDCG@10 | +0.1245 | [+0.1070, +0.1423] |
| Current weighted hybrid | Recall@10 | +0.1109 | [+0.0899, +0.1319] |
| Token overlap | nDCG@10 | +0.2077 | [+0.1903, +0.2256] |
| Token overlap | Recall@10 | +0.1003 | [+0.0812, +0.1202] |
| Hashing cosine | nDCG@10 | +0.2520 | [+0.2283, +0.2765] |
| Hashing cosine | Recall@10 | +0.2295 | [+0.2024, +0.2572] |

## Standalone FTS engine scale results

| Generated corpus | nDCG@10 | Recall@10 | Index time | Index bytes | Warm p50 | Warm p95 |
|---|---:|---:|---:|---:|---:|---:|
| 100,000 passages | 1.0000 | 1.0000 | 0.58 s | 83,525,632 | 53.38 ms | 56.77 ms |
| 1,000,000 passages | 1.0000 | 1.0000 | 7.10 s | 833,110,016 | 568.47 ms | 605.16 ms |

These implementation-ablation stress results use the standalone FTS engine,
not the full `ProductionFTS5System` path. They demonstrate FTS correctness and
expose the local SQLite latency curve; they are not full-stack storage figures
or evidence of internet-scale or hosted-service performance. The generated
corpus is intentionally deterministic and easier than the public datasets.

## Scope and limitations

- BEIR evaluation is passage retrieval. It holds parsing and summarization
  constant and does not measure answer generation quality.
- Deterministic grounding checks are available only on the generated corpus.
  No answer-model run was made because no fixed answer model and credential
  were supplied.
- Cold latency is the first measured query after indexing; warm p50/p95 uses
  the remaining queries. Results are single-machine measurements, not a
  concurrency test.
- Hosted Pinecone and Weaviate comparisons remain out of scope until
  credentials and a spend cap are provided.
- The benchmark supports local architectural claims only. It does not support
  a claim that Librarian outperforms enterprise retrieval systems.

