"""Catalog a SQL database: each table is opened, sampled, and profiled.

Demonstrates the "open the asset, understand its structure, sample the first
records" behavior for relational data. Uses SQLite from the standard library.
"""

import os
import sqlite3
import tempfile

from librarian import Librarian, SQLConnector


def make_sample_db(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE customers (id INTEGER, name TEXT, email TEXT, plan TEXT, mrr REAL)"
    )
    conn.executemany(
        "INSERT INTO customers VALUES (?,?,?,?,?)",
        [
            (i, f"Customer {i}", f"user{i}@example.com",
             "Enterprise" if i % 3 == 0 else "Pro", 49.0 * (i % 5 + 1))
            for i in range(1, 40)
        ],
    )
    conn.execute(
        "CREATE TABLE invoices (id INTEGER, customer_id INTEGER, amount REAL, status TEXT)"
    )
    conn.executemany(
        "INSERT INTO invoices VALUES (?,?,?,?)",
        [(i, i, 99.0, "paid" if i % 2 else "open") for i in range(1, 25)],
    )
    conn.commit()
    conn.close()


def main() -> None:
    workdir = tempfile.mkdtemp(prefix="librarian_sql_")
    dbp = os.path.join(workdir, "shop.db")
    make_sample_db(dbp)

    lib = Librarian.open(os.path.join(workdir, "kb"))
    lib.add_connector(SQLConnector(sqlite_path=dbp, source_id="shopdb", sample_rows=10))
    print("Build:", lib.build())

    for doc in lib.catalog.all_documents():
        profile = doc.metadata["profile"]
        cols = ", ".join(f"{c['name']}:{c['inferred_type']}" for c in profile["columns"])
        print(f"\n{doc.title}")
        print(f"  modality : {profile['modality']}")
        print(f"  columns  : {cols}")

    print("\nQ: which customers are on the enterprise plan?")
    for ev in lib.search("customers on the enterprise plan with email", k=3):
        print(f"   [{ev.score:.3f}] {ev.citation()}")
    lib.close()


if __name__ == "__main__":
    main()
