import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rag.nodes as nodes
import rag.routing as routing
from main import run_question


class Counted:
    def __init__(self, chain, name, counts):
        self.chain, self.name, self.counts = chain, name, counts

    def invoke(self, *args, **kwargs):
        self.counts[self.name] = self.counts.get(self.name, 0) + 1
        return self.chain.invoke(*args, **kwargs)


counts = {}
nodes.retrieval_grader = Counted(nodes.retrieval_grader, "retrieval_grader", counts)
nodes.generation_chain = Counted(nodes.generation_chain, "generation", counts)
routing.question_router = Counted(routing.question_router, "router", counts)
routing.hallucination_grader = Counted(routing.hallucination_grader, "hallucination", counts)
routing.answer_grader = Counted(routing.answer_grader, "answer", counts)

for q in ["What is chain of thought prompting?", "Who won the cricket world cup?"]:
    counts.clear()
    path, _ = run_question(q)
    print(f"\n{q}\n   path : {' -> '.join(path)}\n   calls: {counts}   total={sum(counts.values())}")