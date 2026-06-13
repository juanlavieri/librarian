"""Source connectors: catalog *any* document store, however deep.

A connector knows how to walk one kind of store and yield :class:`IntakeItem`s.
The Librarian treats them uniformly, so the same enrichment + cataloging
pipeline runs over a laptop folder, a website, a SharePoint drive, or a SQL
database. Connectors descend as deep as the store goes -- every branch, every
leaf -- and open each asset so it can be profiled and described.

Built in:

* :class:`FilesystemConnector` -- recursive directory walk (default).
* :class:`WebConnector` -- same-site crawl (needs the ``web`` extra).
* :class:`SQLConnector` -- introspects tables and samples the first N rows of
  each so the catalog understands the *structure* of the data, not just that a
  database exists. Works on SQLite out of the box and on any DB-API connection
  or SQLAlchemy engine you pass in.

Implement the tiny :class:`Connector` protocol to add SharePoint, S3, Notion,
Confluence, a vector of PDFs, or a proprietary store -- nothing downstream
changes.
"""

from __future__ import annotations

from typing import Iterable, Iterator, List, Optional, Protocol

from .ingest import IntakeItem, from_pages, from_path


class Connector(Protocol):
    def items(self) -> Iterator[IntakeItem]:  # pragma: no cover - protocol
        ...


class FilesystemConnector:
    def __init__(
        self,
        path: str,
        *,
        source_id: str = "filesystem",
        recursive: bool = True,
        extensions: Optional[frozenset] = None,
    ) -> None:
        self.path = path
        self.source_id = source_id
        self.recursive = recursive
        self.extensions = extensions

    def items(self) -> Iterator[IntakeItem]:
        yield from from_path(
            self.path,
            source_id=self.source_id,
            recursive=self.recursive,
            extensions=self.extensions,
        )


class WebConnector:
    def __init__(
        self,
        start_url: str,
        *,
        source_id: str = "web",
        max_pages: int = 100,
        max_depth: int = 6,
    ) -> None:
        self.start_url = start_url
        self.source_id = source_id
        self.max_pages = max_pages
        self.max_depth = max_depth

    def items(self) -> Iterator[IntakeItem]:
        from .crawl import crawl_site

        pages = crawl_site(
            self.start_url, max_pages=self.max_pages, max_depth=self.max_depth
        )
        yield from from_pages(pages, source_id=self.source_id)


class SQLConnector:
    """Catalog a relational database by sampling each table.

    For every table the connector reads the column names and the first
    ``sample_rows`` records, renders them as a pipe-delimited grid (which the
    profiler recognizes as tabular), and emits one intake item per table. The
    catalog therefore ends up knowing each table's schema, row sample, and
    inferred subject.
    """

    def __init__(
        self,
        *,
        sqlite_path: Optional[str] = None,
        connection=None,
        engine=None,
        source_id: str = "database",
        sample_rows: int = 10,
        tables: Optional[List[str]] = None,
    ) -> None:
        if not (sqlite_path or connection or engine):
            raise ValueError("Provide sqlite_path, connection, or engine")
        self.sqlite_path = sqlite_path
        self.connection = connection
        self.engine = engine
        self.source_id = source_id
        self.sample_rows = sample_rows
        self.tables = tables

    def _get_cursor_conn(self):
        if self.engine is not None:
            return self.engine.raw_connection()
        if self.connection is not None:
            return self.connection
        import sqlite3

        return sqlite3.connect(self.sqlite_path)

    def _list_tables(self, conn) -> List[str]:
        if self.tables:
            return self.tables
        cur = conn.cursor()
        # SQLite catalog query; override by passing `tables=` for other engines.
        try:
            cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
            return [r[0] for r in cur.fetchall()]
        except Exception:
            return []

    def items(self) -> Iterator[IntakeItem]:
        owns = self.engine is not None or self.connection is None
        conn = self._get_cursor_conn()
        try:
            for table in self._list_tables(conn):
                item = self._sample_table(conn, table)
                if item is not None:
                    yield item
        finally:
            if owns:
                try:
                    conn.close()
                except Exception:
                    pass

    def _sample_table(self, conn, table: str) -> Optional[IntakeItem]:
        import csv
        import io

        cur = conn.cursor()
        try:
            cur.execute(f'SELECT * FROM "{table}" LIMIT {int(self.sample_rows)}')
            rows = cur.fetchall()
        except Exception:
            return None
        columns = [d[0] for d in (cur.description or [])]
        if not columns:
            return None
        row_count = self._count(conn, table)
        # Emit proper CSV (handles quoting) so the standard reader/profiler
        # recognize it as tabular; the true row count rides in the title, which
        # is indexed alongside the summary.
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([str(c) for c in columns])
        for row in rows:
            writer.writerow(["" if v is None else str(v) for v in row])
        count_label = f" (~{row_count} rows)" if row_count is not None else ""
        return IntakeItem(
            uri=f"sql://{self.source_id}/{table}",
            name=f"{table}.csv",  # .csv => profiled as a table
            data=buf.getvalue().encode("utf-8"),
            source_id=self.source_id,
            title=f"Table: {table}{count_label}",
            media_type="csv",
            metadata={
                "kind": "sql_table",
                "table": table,
                "row_count": row_count,
                "sampled_rows": len(rows),
                "columns": columns,
            },
        )

    @staticmethod
    def _count(conn, table: str) -> Optional[int]:
        try:
            cur = conn.cursor()
            cur.execute(f'SELECT COUNT(*) FROM "{table}"')
            return int(cur.fetchone()[0])
        except Exception:
            return None


def items_from(sources: Iterable[Connector]) -> Iterator[IntakeItem]:
    for connector in sources:
        yield from connector.items()
