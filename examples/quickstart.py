"""Quickstart: build a catalog from a local folder and search it.

Runs with zero dependencies and no API keys (offline hashing embedder +
extractive summarizer). For real semantics, set OPENAI_API_KEY and pass
``embedding_provider="openai"`` / ``summarizer_provider="openai"``.
"""

import os
import tempfile

from librarian import Librarian


def make_sample_docs(root: str) -> str:
    docs = os.path.join(root, "docs")
    os.makedirs(os.path.join(docs, "billing"), exist_ok=True)
    os.makedirs(os.path.join(docs, "engineering"), exist_ok=True)
    with open(os.path.join(docs, "billing", "pricing.md"), "w") as fh:
        fh.write(
            "# Pricing\n\nThe Pro plan costs $49 per month. The Enterprise plan "
            "is custom-priced. Invoices are generated on the first of each month "
            "and emailed to the account owner.\n"
        )
    with open(os.path.join(docs, "billing", "refunds.txt"), "w") as fh:
        fh.write(
            "Refund policy: customers may request a refund within 30 days of "
            "purchase. Approved refunds are processed within 5 business days back "
            "to the original payment method.\n"
        )
    with open(os.path.join(docs, "engineering", "architecture.md"), "w") as fh:
        fh.write(
            "# Architecture\n\nThe service is a FastAPI application backed by a "
            "Postgres database and a Redis cache. Background jobs run on a worker "
            "fleet. Deployments are containerized.\n"
        )
    return docs


def main() -> None:
    workdir = tempfile.mkdtemp(prefix="librarian_quickstart_")
    docs = make_sample_docs(workdir)

    lib = Librarian.open(os.path.join(workdir, "kb"))
    lib.add_path(docs, source_id="handbook")
    stats = lib.build()
    print("Build:", stats)
    print("Stats:", lib.stats())

    for question in [
        "how much does the pro plan cost?",
        "what is the refund window?",
        "which database does the service use?",
    ]:
        print(f"\nQ: {question}")
        for ev in lib.search(question, k=3):
            print(f"   [{ev.score:.3f}] {ev.doc_type:14} {ev.citation()}")

    print("\n--- Injectable context block ---")
    print(lib.context("how do refunds work?", k=2))
    lib.close()


if __name__ == "__main__":
    main()
