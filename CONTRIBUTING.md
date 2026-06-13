# Contributing to Librarian

Thanks for your interest in improving Librarian. This project aims to be a
solid, reusable knowledge layer that anyone can build on, so contributions that
keep the core small, dependency-light, and backend-agnostic are especially
welcome.

## Development setup

```bash
git clone <your-fork-url>
cd librarian
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The package uses a `src/` layout. Tests add `src/` to the path automatically
(see `tests/conftest.py`), so `pytest` works without installation too.

## Guiding principles

1. **The core has zero required dependencies.** Anything heavier (numpy, pdf
   readers, openai, faiss, requests) must be an optional extra, imported lazily,
   and degrade gracefully when absent. Never let a missing optional dependency
   crash a build.
2. **Respect the contracts.** The dataclasses in `models.py` and the protocols
   (`Reader`, `Connector`, `Embedder`, `Summarizer`, `Catalog`, `VectorStore`)
   are the seams. New backends implement a protocol; they don't change call
   sites.
3. **Context and metadata, recursively.** Enrichment is the differentiator. New
   readers/connectors should surface as much structure and metadata as they can.
4. **Provenance is mandatory.** Anything retrievable must carry enough to cite
   it (title, uri, location).
5. **Determinism by default.** Offline defaults should produce stable output.

## Adding things

- **A connector** (new document store): implement `Connector.items()` yielding
  `IntakeItem`s. See `connectors.py`.
- **A reader** (new file type): implement `Reader.parse()` returning located
  `Block`s, then `register_reader(...)`. See `readers/base.py`.
- **A backend** (embedder / summarizer / catalog / vector store): implement the
  matching protocol and pass an instance into `Librarian(...)`, or wire it into
  `config.py` + the `_default_*` resolvers.

## Tests

Please add tests for new functionality. Keep them offline and fast — use the
default hashing embedder and `tmp_path`. Network/credentialed integrations
should be skipped when their optional dependency or key is unavailable.

## Style

- Standard library + type hints; keep functions small and documented at the
  module level with *why*, not *what*.
- Run `pytest` before opening a PR.

## License

By contributing you agree your contributions are licensed under the project's
[Apache 2.0](LICENSE) license.
