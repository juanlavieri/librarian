# Librarian

**A local, evidence-oriented knowledge layer for AI systems.**

Librarian turns a document store — a folder on your laptop, a website, or a SQL
database out of the box, and anything else (SharePoint, S3, Notion, …) through a
small connector interface — into a context-rich, logically organized
**catalog**, and gives you a hybrid **retriever**, designed to be driven by your
agent, that returns passage-level evidence with source references.

It is plug-and-play and runs with **zero required dependencies and no API
keys**. `pip install`, point it at your data, and search.

```python
from librarian import Librarian

lib = Librarian.open("./kb")
lib.add_path("./docs")        # or .add_url(...), .add_connector(SQLConnector(...))
lib.build()                   # crawl → profile → summarize → chunk → organize → index

for ev in lib.search("what is our refund window?"):
    print(ev.score, ev.citation())
```

---

## Positioning and evidence standard

Librarian is currently optimized for private retrieval across the files and
operational context of an individual or small team. It is **not presented as a
proven replacement for Pinecone, Weaviate, or an enterprise retrieval stack**.
No controlled head-to-head benchmark against those systems has been published.

The architectural focus is hierarchy and evidence: retrieval records retain
document identity, version, source, location, metadata, and structural roll-ups.
A retrieval score estimates relevance; it does not establish authority and is
not the source of truth. The SQLite catalog remains canonical for the metadata
managed by this package.

This open-source release currently implements:

- a local SQLite metadata catalog plus an in-memory FTS5/BM25 retrieval index;
- summary records and passage-level document chunks;
- deterministic hashing embeddings by default, with OpenAI embeddings optional;
- cosine-similarity semantic ranking;
- FTS5/BM25 retrieval by default, promoted by the published local benchmark;
- legacy lexical token overlap and configurable weighted or reciprocal-rank
  fusion for controlled comparisons;
- structural folder and section roll-ups in the same retrieval index; and
- local JSON persistence by default, with an optional FAISS backend.

When hybrid retrieval is selected, semantic-provider failures automatically
fall back to lexical retrieval. The package does not implement an authorization
policy engine; integrating applications and connectors must enforce source
permissions before content is indexed or returned.

The local benchmark measured FTS5/BM25 against the previous weighted hybrid,
token overlap, hashing cosine, MiniLM, and RRF on BEIR SciFact, BEIR NFCorpus,
and a generated hierarchical/versioned corpus. FTS5/BM25 improved pooled
nDCG@10 by 0.1245 over the previous hybrid (paired bootstrap 95% CI
[0.1070, 0.1423]) without an individual-dataset regression. See the
[methodology](docs/BENCHMARKING.md) and [full results](docs/BENCHMARK_RESULTS.md).
This is not a hosted-service or enterprise comparison.

## Thesis: AI systems need a knowledge layer

The single most reliable way to make an AI system more useful is to give it the
*right* context at the *right* moment. Yet the layer responsible for that —
ingesting information, understanding it, organizing it, keeping it current, and
serving it back on demand — is almost always rebuilt from scratch, badly, by
every team that needs it.

The dominant pattern, naive vector-only RAG, looks deceptively complete:

1. split documents into chunks,
2. embed the chunks,
3. retrieve the top-k by cosine similarity,
4. stuff them into a prompt.

This can be useful, but it discards signals that matter in hierarchical,
heterogeneous repositories. A chunk-only vector design can:

- **flattens structure** — folders, tables, document relationships, and project
  context all disappear into an undifferentiated soup of fragments;
- **fragments meaning** — a chunk pulled from the middle of a document ("it grew
  12% year over year") is uninterpretable without the context it was severed
  from;
- **is opaque** — there's no good answer to *why* a chunk was retrieved, or
  *where* in the corpus the answer lives;
- **omit authority signals** — similarity alone does not establish whether a
  source is current, permitted, authoritative, or the system of record.

These limitations motivate preserving structure and provenance alongside
retrieval scores. They do not, by themselves, prove better retrieval quality.

### The Librarian model

Think about how a great research library actually works. It is **two systems**:

1. **A catalog.** Every item on every shelf has been opened, understood,
   and described. The catalog knows what each item is, what it's about, and
   where it sits in the structure — and it summarizes whole sections and
   collections, not just individual books.
2. **A librarian.** A person (now: an agent) who knows how to *navigate* that
   catalog. You don't recite keywords at them; you describe what you need, and
   they walk you to the right shelf, the right book, the right page — fast,
   even in the largest library on earth.

Librarian is the software embodiment of both halves. It reframes retrieval from
a *similarity-search problem* into a **knowledge-navigation problem**: not
"which chunks are nearest in embedding space?" but "how would a knowledgeable
expert locate, interpret, and explain this?"

### What makes it different: context and metadata, all the way down

The novel core of Librarian is that **everything is enriched with context and
metadata, recursively, at every level of the tree.**

- **Deep, recursive cataloging.** A connector descends as deep as the store
  goes — every branch, every leaf. It doesn't just list assets; it **opens
  each one**. It works out whether an asset is prose, a table, or structured
  data. For a table it reads the header, samples the first rows, and infers each
  column's type and the row count, so the catalog *knows* a file is "a list of
  customers with an email column and ~24 rows," not just "bytes."
- **Metadata bubbles up the tree.** Each asset's profile is attributed to its
  folder, and to every ancestor folder, recursively. Each folder is then
  summarized from the documents and sub-folder roll-ups beneath it. Leaf-level
  understanding propagates all the way to the root, so you can ask "what's in
  this whole area?" at *any* altitude and get a real answer.
- **Summaries before chunks.** Librarian embeds clean, human-readable summaries
  as the primary retrieval unit. Summaries carry stronger semantic signal, cost
  far less to store and search, and stay inspectable. Chunks are a *fallback*,
  used only when a question genuinely needs depth.
- **Context-enriched chunks.** Before a chunk is embedded, Librarian prepends a
  compact context header — the document, its location, its inferred subject, and
  the nearest heading. So "it grew 12% year over year" becomes a unit that knows
  *what* grew and *which* document it's from. This is intended to make retrieved
  passages easier to interpret and ground.
- **Hybrid, structure-aware retrieval.** The read path blends complementary
  signals the way an expert does: semantic similarity, lexical term overlap
  (over titles, paths, and tags), and structural roll-ups — preferring current
  editions and summaries, then deepening into chunks on demand.

The design goal is to make relevance inspectable rather than leaving it to a
single distance metric. Whether that improves retrieval or answer quality is an
empirical question for the benchmark suite.

### A foundation to build on

Heavy, general-purpose foundations change what everyone else can build. When the
hard, shared substrate of a problem becomes a solid, open, reusable layer, an
entire ecosystem grows on top of it. Knowledge management for AI is exactly that
kind of shared substrate — every serious AI system needs it, and almost no one
should be reinventing it.

Librarian is built as a modular local foundation: backends are swappable,
summary-first retrieval can deepen into passages, and index implementations are
pluggable. Claims about latency, cost, retrieval quality, or behavior at larger
scales are intentionally deferred until measured.

---

## How it works

```
                          ┌──────────────────────── THE CATALOG (write path) ───────────────────────┐
  any document store ──▶  │  connect → descend deep → open & profile each asset → summarize →        │
  (files, web, SQL,       │  context-enrich chunks → recursively roll up metadata → organize         │
   SharePoint, S3, …)     │  (virtual sections) → embed → catalog (SQL) + search index (vectors)     │
                          └──────────────────────────────────────────────────────────────────────────┘
                                                          │
  agent / app  ──▶  ask a question  ──▶  ┌──────────── THE LIBRARIAN (read path) ─────────────┐
                                         │  hybrid retrieve (semantic + lexical + structural +  │
                                         │  path metadata) → prefer current summaries → deepen  │
                                         │  into context-rich chunks when needed → cite         │
                                         └──────────────────────────────────────────────────────┘
                                                          │
                                          passage-level, citation-ready evidence
```

### The pipeline, stage by stage

| Stage | Module | What it does |
|---|---|---|
| **Connect** | `connectors.py` | Walk any store as deep as it goes. Built-in: filesystem, web crawl, SQL (samples each table). Pluggable for SharePoint, S3, Notion, … |
| **Read** | `readers/` | Turn bytes into located blocks (`p.12`, `slide 8`, `Sheet1`). Text/CSV/JSON/HTML are dependency-free; PDF/DOCX/PPTX/XLSX/OCR are optional. |
| **Profile** | `enrich.py` | Open each asset; detect modality; infer table schema + sample rows; extract topics; write a one-line "what's inside" description. |
| **Summarize** | `summarize.py` | Summary-first understanding. Offline extractive by default; OpenAI optional. |
| **Chunk** | `chunking.py` + `context.py` | Heading/location-aware chunking with overlap, then prepend a context header to every chunk. |
| **Organize** | `rollups.py` | Recursively roll up metadata into parent folders; optional virtual "sections" (shelves) that never move the source bytes. |
| **Catalog** | `catalog/` | Canonical source of truth: documents, immutable versions, chunks, sections, membership. SQLite by default. |
| **Index** | `vectorstore/` | Denormalized, searchable records + embeddings. Local pure-Python store by default; FAISS optional. |
| **Retrieve** | `retrieval.py` | Hybrid, structure-aware, summary-first with chunk fallback. Returns `Evidence` with provenance. |
| **Serve** | `tool.py`, `memory.py` | Agent tool adapter (OpenAI / Anthropic / LangChain) + short-term conversational memory. |

### Architectural emphasis (not benchmark results)

Librarian preserves recursive roll-ups, per-asset metadata, contextual chunk
headers, source locations, immutable versions, and current-edition filtering.
These are implementation characteristics—not evidence that it retrieves more
accurately, responds faster, costs less, or produces better grounded answers
than another system.

---

## Install

```bash
pip install librarian-ai            # core, zero dependencies

pip install "librarian-ai[fast]"        # numpy-accelerated local search
pip install "librarian-ai[documents]"   # PDF / DOCX / PPTX / XLSX readers
pip install "librarian-ai[web]"         # website crawling connector
pip install "librarian-ai[openai]"      # OpenAI embeddings + summaries
pip install "librarian-ai[faiss]"       # FAISS vector backend (scale-out)
pip install "librarian-ai[all]"         # everything
```

> **Package name:** install with `pip install librarian-ai`; import it as
> `librarian` (`from librarian import Librarian`). See
> [Package name & history](#package-name--history) for why the distribution is
> named `librarian-ai`.

## Quickstart

### Python

```python
from librarian import Librarian

lib = Librarian.open("./kb")
lib.add_path("./docs", source_id="docs")
print(lib.build())            # {'indexed': 42, 'skipped': 0, 'chunks': 318, ...}

# Search → structured, citation-ready evidence
for ev in lib.search("how do refunds work?", k=5):
    print(f"{ev.score:.3f}  {ev.doc_type:14}  {ev.citation()}")

# Or get a ready-to-inject context block with inline citations
context = lib.context("how do refunds work?")
```

### As an agent tool

The Librarian's read path drops into any agent runtime as a function tool:

```python
tool = lib.as_tool()

tool.openai_schema()             # OpenAI Chat Completions  (tools=[...])
tool.openai_responses_schema()   # OpenAI Responses API
tool.anthropic_schema()          # Anthropic Messages API   (tools=[...])
tool.as_langchain_tool()         # LangChain StructuredTool

# Dispatch when the model calls the tool:
result = tool.run("refund window")        # -> {"evidence": [...]}
payload = tool.run_json("refund window")  # same, JSON-encoded for the tool message
```

### Command line

```bash
librarian --root ./kb index ./docs --source handbook
librarian --root ./kb index https://example.com --source site --max-pages 50
librarian --root ./kb search "how do I set up the VPN" -k 5
librarian --root ./kb context "vpn setup"
librarian --root ./kb stats
```

### Cataloging a database

```python
from librarian import Librarian, SQLConnector

lib = Librarian.open("./kb")
lib.add_connector(SQLConnector(sqlite_path="shop.db", source_id="shopdb", sample_rows=10))
lib.build()
# Each table is profiled: columns, inferred types, sample rows, and row count.
```

---

## Plug-and-play and malleable: every layer is swappable

Sensible, offline defaults; production backends behind a one-line change.

```python
from librarian import Librarian, LibrarianConfig

cfg = LibrarianConfig(
    root="./kb",
    embedding_provider="openai",      # default: "hashing" (offline, no key)
    summarizer_provider="openai",     # default: "extractive" (offline)
    vector_backend="faiss",           # default: "local" (pure-Python/numpy)
    catalog_backend="sqlite",         # default
)
lib = Librarian(cfg)
```

You can also inject your own components directly:

```python
from librarian import Librarian

lib = Librarian(
    embedder=MyEmbedder(),            # implements embed() / embed_one()
    summarizer=MySummarizer(),        # implements summarize()
    catalog=MyCatalog(),              # Postgres, Snowflake, …
    store=MyVectorStore(),            # pgvector, Pinecone, Qdrant, …
)
```

Add support for a new store or file format without forking:

```python
from librarian import register_reader, FilesystemConnector
# Implement the small Connector / Reader protocols (see connectors.py, readers/base.py).
```

### Extension points

| Want to… | Implement | Default |
|---|---|---|
| Catalog a new store (SharePoint, S3, Notion) | `Connector` | filesystem / web / SQL |
| Support a new file type | `Reader` + `register_reader` | text, csv, json, html, pdf, office, image |
| Use real embeddings | `Embedder` | hashing (offline) |
| Use better summaries | `Summarizer` | extractive (offline) |
| Scale the index | `VectorStore` | local / faiss |
| Change the metadata store | `Catalog` | SQLite |

---

## Design principles

- **Two-part system.** A well-organized catalog *and* an effective librarian.
  Neither alone is enough.
- **Context and metadata, recursively.** Enrichment at the asset, chunk, folder,
  and collection level — propagated up the tree.
- **Summary-first, deepen on demand.** Search summaries first; use chunks when
  the query or score threshold calls for more detail.
- **Stable identity, immutable versions, virtual organization.** Nothing is ever
  moved or renamed in storage; only metadata and membership change. The same
  source always maps to the same `doc_id`; new content always makes a new
  `version_id`.
- **Provenance is mandatory.** Every result carries enough to cite it.
- **Backend-agnostic.** The data model is the contract; storage is an
  implementation detail.
- **Plug-and-play locally.** The default configuration runs without external
  services; larger deployments require an appropriate custom backend and
  measurement.

## Status & roadmap

Current development version — core catalog + benchmark-promoted FTS5/BM25
retrieval, offline defaults, filesystem/web/SQL connectors, OpenAI + FAISS
integrations, agent tool + CLI.

Planned: agentic organization (LLM-proposed sections/merges), incremental delta
sync, more connectors (SharePoint/S3/Notion/Confluence), pgvector/Qdrant stores,
hosted retrieval comparisons, and async ingestion. The reproducible
[evaluation harness](benchmarks/README.md) and [results](docs/BENCHMARK_RESULTS.md)
are included in this repository.

## Package name & history

- **Install:** `pip install librarian-ai`
- **Import:** `import librarian` / `from librarian import Librarian`
- **Source:** https://github.com/juanlavieri/librarian

The PyPI distribution is named **`librarian-ai`**. The Python import package is
`librarian` (the shorter, intuitive name to type in code); the distribution uses
the `-ai` suffix because the bare `librarian` name is already taken on PyPI.

> ⚠️ A package named **`librarian-kb`** (version `0.1.0`) also exists on PyPI. It
> was an earlier release of this project published from an account that is no
> longer accessible. **It is not maintained — do not use it.** The canonical,
> maintained package is **`librarian-ai`**.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues and PRs welcome.

## License

Copyright 2026 Juan Lavieri. Licensed under [Apache 2.0](LICENSE) (see also
[NOTICE](NOTICE)).
