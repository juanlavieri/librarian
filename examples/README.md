# Examples

Runnable, dependency-free examples. From the repo root:

```bash
pip install -e .          # or: pip install librarian-kb
python examples/quickstart.py
python examples/sql_catalog.py
python examples/agent_tool_openai.py     # prints the wiring; runs the LLM only if OPENAI_API_KEY is set
```

| File | Shows |
|---|---|
| `quickstart.py` | Build a catalog from a local folder, search it, print a context block. |
| `sql_catalog.py` | Catalog a SQLite database — each table profiled with schema + sample rows. |
| `agent_tool_openai.py` | Expose the Librarian as an OpenAI function tool and run an agent loop. |
