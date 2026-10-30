import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document
from langgraph.graph import END

from rag.core.config import GENERATE, NO_ANSWER, RETRIEVE, WEBSEARCH
from rag.routing import decide_after_generation, decide_after_grading, route_question

DOCS = [Document(page_content="Agent memory has short-term and long-term parts.")]
Q = "What is agent memory?"


def check(label, got, expected):
    print(f"{'ok  ' if got == expected else 'MISS'} {label:42} expected={expected:14} got={got}")


# A: entry (2 LLM calls)
check("on-topic question", route_question({"question": Q}), RETRIEVE)
check("off-topic question", route_question({"question": "Who won the cricket world cup?"}), WEBSEARCH)

# B: after grading (no LLM)
check("a chunk was dropped", decide_after_grading({"web_search": True}), WEBSEARCH)
check("all chunks kept", decide_after_grading({"web_search": False}), GENERATE)

# C: after generation (LLM judges)
base = {"question": Q, "documents": DOCS}
check("marker, attempt 1", decide_after_generation({**base, "generation": NO_ANSWER, "attempts": 1}), WEBSEARCH)
check("marker, attempt 2 (cap)", decide_after_generation({**base, "generation": NO_ANSWER, "attempts": 2}), END)
check("good answer", decide_after_generation(
    {**base, "generation": "Agent memory has short-term and long-term parts.", "attempts": 1}), END)
check("invented answer, attempt 1", decide_after_generation(
    {**base, "generation": "Agents store memory in quantum crystals.", "attempts": 1}), GENERATE)
check("invented answer, attempt 2 (cap)", decide_after_generation(
    {**base, "generation": "Agents store memory in quantum crystals.", "attempts": 2}), END)