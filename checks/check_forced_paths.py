import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rag.nodes as nodes
from main import run_question
from rag.core.config import NO_ANSWER

GOOD = "Chain of thought prompting asks a model to reason step by step."
BAD = "Chain of thought prompting stores memory in quantum crystals."

class FakeGeneration:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    def invoke(self, inputs):
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return reply


Q = "What is chain of thought prompting?"
CASES = [
    ("refusal, then a real answer", [NO_ANSWER, GOOD]),
    ("invented answer, then a real one", [BAD, GOOD]),
    ("invented answer every time", [BAD]),
]



real = nodes.generation_chain
for label, replies in CASES:
    nodes.generation_chain = FakeGeneration(replies)
    path, state = run_question(Q)
    print(f"\n{label}")
    print("   path    :", " -> ".join(path))
    print("   attempts:", state["attempts"])
    print("   final   :", state["generation"][:70])
nodes.generation_chain = real