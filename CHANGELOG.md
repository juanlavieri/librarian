# Changelog

All notable changes to this project are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/), and the project
adheres to [Semantic Versioning](https://semver.org/).

## [0.1.1] - 2026-06-23

### Fixed
- **Archived-version retrieval now works.** Superseding a document archives its
  prior index records (`is_current=false`) instead of deleting them, so
  `include_archived=True` can retrieve earlier editions while default searches
  return only the current one.
- **Lexical search matches URIs/paths**, so filename and path queries resolve as
  documented.
- **Resilient builds.** A failing connector (e.g. a missing optional dependency
  or an unreachable store) is counted as failed and skipped instead of aborting
  the whole build.
- **Robust archival-capability detection for custom vector stores.** The
  Librarian now correctly distinguishes a real `archive_doc` /
  `delete_by_doc_version` implementation -- including ones provided via instance
  attributes or `__getattr__` delegation -- from an inherited no-op `Protocol`
  stub, and falls back to `delete_by_doc` only when the helpers are genuinely
  absent.

### Changed
- Documentation tightened so claims match behavior (built-in vs. pluggable
  connectors; retriever framing).

## [0.1.0] - 2026-06-13

### Added
- Initial release: the two-part knowledge layer (context-rich catalog + hybrid
  retriever), filesystem/web/SQL connectors, asset profiling, summary-first
  indexing, contextual chunks, recursive folder roll-ups, virtual sections,
  conversational memory, an agent tool adapter, and a CLI. Offline defaults with
  optional OpenAI and FAISS backends.
