from librarian.memory import ConversationMemory


def test_memory_keeps_recent_turns():
    mem = ConversationMemory(max_turns=4)
    for i in range(6):
        mem.add_user(f"question {i}")
        mem.add_assistant(f"answer {i}")
    assert len(mem.recent_turns()) <= 4
    # older turns compressed into the summary
    assert mem.summary()


def test_context_query_includes_query():
    mem = ConversationMemory(max_turns=4)
    mem.add_user("tell me about the valve document")
    q = mem.context_query("what pressure rating?")
    assert "pressure" in q
    assert "valve" in q


def test_clear():
    mem = ConversationMemory()
    mem.add_user("x")
    mem.clear()
    assert not mem.recent_turns()
    assert mem.summary() == ""
