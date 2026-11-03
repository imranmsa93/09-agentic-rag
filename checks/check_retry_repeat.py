import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document

from rag.nodes import generate_node

state = {
    "question": "What is chain of thought prompting?",
    "documents": [Document(page_content=(
        "Chain of thought prompting asks a model to write out its reasoning "
        "steps before giving a final answer."
    ))],
    "attempts": 0,
}

first = generate_node(state)["generation"]
second = generate_node(state)["generation"]
print("identical:", first == second)
print("1:", first)
print("2:", second)