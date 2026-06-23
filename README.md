# Librarian

**A foundational knowledge layer for AI systems.**

Librarian turns a document store — a folder on your laptop, a website, or a SQL
database out of the box, and anything else (SharePoint, S3, Notion, …) through a
small connector interface — into a context-rich, logically organized
**catalog**, and gives you a hybrid **retriever**, designed to be driven by your
agent, that finds the right information inside it, fast, with citations.

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

## Thesis: AI doesn't have a model problem, it has a knowledge problem

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

This works in demos and breaks in the real world, because real knowledge bases
are **large, hierarchical, heterogeneous, and messy**. Vector-only RAG:

- **flattens structure** — folders, tables, document relationships, and project
  context all disappear into an undifferentiated soup of fragments;
- **fragments meaning** — a chunk pulled from the middle of a document ("it grew
  12% year over year") is uninterpretable without the context it was severed
  from;
- **is opaque** — there's no good answer to *why* a chunk was retrieved, or
  *where* in the corpus the answer lives;
- **degrades with scale** — the bigger and messier the corpus, the more
  irrelevant neighbors crowd the top-k, and the more the model hallucinates.

The result is a system that retrieves *plausible* text instead of *relevant*
text, and confidently fills the gaps with fiction.

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
  *what* grew and *which* document it's from. This is what curbs hallucination
  and sharpens precision.
- **Hybrid, structure-aware retrieval.** The read path blends complementary
  signals the way an expert does: semantic similarity, lexical term overlap
  (over titles, paths, and tags), and structural roll-ups — preferring current
  editions and summaries, then deepening into chunks on demand.

The effect: as the corpus grows, the system stays *highly relevant* to the
specific thing being asked, because relevance is engineered into the catalog,
not left to a single distance metric.

### A foundation to build on

Heavy, general-purpose foundations change what everyone else can build. When the
hard, shared substrate of a problem becomes a solid, open, reusable layer, an
entire ecosystem grows on top of it. Knowledge management for AI is exactly that
kind of shared substrate — every serious AI system needs it, and almost no one
should be reinventing it.

Librarian is built to be that layer: **malleable** (swap any backend),
**fast** (summary-first, lazy deepening, pluggable ANN indexes), and **strong
enough to build on** regardless of scale. It is, in effect, a new kind of search
engine — one designed for AI agents rather than humans typing queries into a
box.

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
                                         │  direct path) → prefer current summaries → deepen    │
                                         │  into context-rich chunks when needed → cite         │
                                         └──────────────────────────────────────────────────────┘
                                                          │
                                          highly relevant, citation-ready evidence
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

### Why it outperforms vector-only RAG

| Dimension | Vector-only RAG | Librarian |
|---|---|---|
| Hierarchy awareness | ✗ | ✓ (recursive roll-ups) |
| Per-asset metadata | ✗ | ✓ (modality, schema, topics) |
| Chunk interpretability | low | ✓ (contextual headers) |
| Explainability / provenance | low | ✓ (citations + locations) |
| Structural ("where is X?") queries | ✗ | ✓ |
| Versioning / current-edition bias | rare | ✓ (immutable versions) |
| Behavior on large, messy corpora | degrades | strong |
| Hallucination pressure | high | reduced |

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
- **Summary-first, deepen on demand.** Cheaper, cleaner, faster; chunks only
  when the question needs them.
- **Stable identity, immutable versions, virtual organization.** Nothing is ever
  moved or renamed in storage; only metadata and membership change. The same
  source always maps to the same `doc_id`; new content always makes a new
  `version_id`.
- **Provenance is mandatory.** Every result carries enough to cite it.
- **Backend-agnostic.** The data model is the contract; storage is an
  implementation detail.
- **Plug-and-play.** Works the instant it's installed; scales when you ask it to.

## Status & roadmap

`0.1.0` — core catalog + retrieval, offline defaults, filesystem/web/SQL
connectors, OpenAI + FAISS integrations, agent tool + CLI.

Planned: agentic organization (LLM-proposed sections/merges), incremental delta
sync, more connectors (SharePoint/S3/Notion/Confluence), pgvector/Qdrant stores,
evaluation harness, and async ingestion.

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
