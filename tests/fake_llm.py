"""A scripted stand-in for the Groq client, so the whole pipeline can be tested
without an API key. It also behaves like the real API in one strict way: it
REJECTS a request if any earlier tool call has no matching tool reply."""
import json
from types import SimpleNamespace


class _Msg:
    def __init__(self, content=None, calls=None):
        self.content = content
        self.tool_calls = [
            SimpleNamespace(id=cid, function=SimpleNamespace(name=n, arguments=json.dumps(a)))
            for cid, n, a in (calls or [])
        ] or None
        self._calls = calls or []

    def model_dump(self, exclude_unset=True):
        d = {"role": "assistant", "content": self.content}
        if self._calls:
            d["tool_calls"] = [{"id": c, "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
                               for c, n, a in self._calls]
        return d


def _check_tool_replies(messages):
    answered = {m["tool_call_id"] for m in messages if m.get("role") == "tool"}
    for m in messages:
        for tc in m.get("tool_calls") or []:
            if tc["id"] not in answered:
                raise RuntimeError(f"400: tool_call_id {tc['id']} has no tool response")


class FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.n = 0

    def _id(self):
        self.n += 1
        return f"call_{self.n}"

    def _create(self, model, messages, tools=None, **kw):
        _check_tool_replies(messages)
        q = messages[1]["content"].lower()
        last = messages[-1]
        tool_msgs = [m for m in messages if m["role"] == "tool"]

        # ---- after tool output(s): final answer echoes what came back ------
        if tool_msgs:
            out = tool_msgs[-1]["content"]
            if "user_response" in out:                      # clarification answered -> now query
                region = json.loads(out)["user_response"].strip()
                return SimpleNamespace(choices=[SimpleNamespace(message=_Msg(calls=[(self._id(), "run_query", {
                    "query": f"SELECT SUM(amount) AS total FROM uploaded_data WHERE region = '{region}'"})]))])
            return SimpleNamespace(choices=[SimpleNamespace(message=_Msg(content="RESULT: " + out[:700]))])

        # ---- first turn: decide what the "model" does -----------------------
        def call(sql):
            return SimpleNamespace(choices=[SimpleNamespace(message=_Msg(calls=[(self._id(), "run_query", {"query": sql})]))])

        if "joke" in q:
            return SimpleNamespace(choices=[SimpleNamespace(message=_Msg(content="I can only answer questions about this dataset."))])
        if "ambiguous" in q:
            return SimpleNamespace(choices=[SimpleNamespace(message=_Msg(calls=[(self._id(), "ask_user_clarification",
                                                                                 {"question": "Which region do you mean?"})]))])
        if "by region" in q:
            return call("SELECT region, ROUND(SUM(amount),2) AS total, COUNT(*) AS n FROM uploaded_data GROUP BY region ORDER BY total DESC")
        if "how many" in q:
            return call("SELECT COUNT(*) AS n FROM uploaded_data")
        if "top customers" in q:
            return call("SELECT customer_id, SUM(amount) AS total FROM uploaded_data GROUP BY customer_id ORDER BY total DESC LIMIT 5")
        if "drop" in q:
            return call("SELECT 1 LIMIT 1; DROP TABLE uploaded_data")
        if "readfile" in q:
            return call("SELECT * FROM read_csv('/etc/passwd')")
        if "everything" in q:
            return call("SELECT * FROM uploaded_data")
        if "slow" in q:
            return call("SELECT COUNT(*) FROM uploaded_data a, uploaded_data b, uploaded_data c WHERE a.amount + b.amount > c.amount * 3")
        if "two calls" in q:
            return SimpleNamespace(choices=[SimpleNamespace(message=_Msg(calls=[
                (self._id(), "run_query", {"query": "SELECT COUNT(*) AS n FROM uploaded_data"}),
                (self._id(), "run_query", {"query": "SELECT MAX(amount) AS m FROM uploaded_data"})]))])
        return SimpleNamespace(choices=[SimpleNamespace(message=_Msg(content="(fake model) no scripted behaviour for this question"))])
