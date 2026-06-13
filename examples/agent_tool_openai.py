"""Expose the Librarian as an OpenAI function tool and run an agent loop.

The Librarian itself runs fully offline. This example only calls the OpenAI API
for the chat model that *uses* the tool; if OPENAI_API_KEY is unset it prints
the tool schema + a direct retrieval result and exits, so it always runs.
"""

import json
import os
import tempfile

from librarian import Librarian


def build_kb() -> Librarian:
    workdir = tempfile.mkdtemp(prefix="librarian_agent_")
    docs = os.path.join(workdir, "docs")
    os.makedirs(docs, exist_ok=True)
    with open(os.path.join(docs, "support.md"), "w") as fh:
        fh.write(
            "# Support\n\nSupport hours are 9am-6pm Eastern, Monday to Friday. "
            "Priority customers get 24/7 support. Escalations go to the on-call "
            "engineer via the incident channel.\n"
        )
    with open(os.path.join(docs, "security.md"), "w") as fh:
        fh.write(
            "# Security\n\nAll data is encrypted at rest and in transit. We are "
            "SOC 2 Type II certified. Access requires SSO and hardware MFA.\n"
        )
    lib = Librarian.open(os.path.join(workdir, "kb"))
    lib.add_path(docs, source_id="kb")
    lib.build()
    return lib


def main() -> None:
    lib = build_kb()
    tool = lib.as_tool()

    print("Tool schema passed to the model:")
    print(json.dumps(tool.openai_schema(), indent=2))

    question = "What are our support hours and are we SOC 2 certified?"

    if not os.getenv("OPENAI_API_KEY"):
        print("\n[OPENAI_API_KEY not set] Direct retrieval result:")
        print(json.dumps(tool.run(question, k=4), indent=2, default=str))
        lib.close()
        return

    from openai import OpenAI

    client = OpenAI()
    messages = [{"role": "user", "content": question}]
    resp = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, tools=[tool.openai_schema()]
    )
    msg = resp.choices[0].message
    if msg.tool_calls:
        messages.append(msg)
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            result = tool.run_json(args["query"], k=args.get("k", 6))
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result}
            )
        final = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        print("\nAnswer:\n", final.choices[0].message.content)
    else:
        print("\nAnswer:\n", msg.content)
    lib.close()


if __name__ == "__main__":
    main()
