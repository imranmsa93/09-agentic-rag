import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.chains.chains import (
    answer_grader, hallucination_grader, question_router, retrieval_grader,
)

DOCS = "Agent memory has short-term and long-term parts."
Q = "What is agent memory?"

CASES = [
    ("router", question_router, {"question": Q}, "vectorstore"),
    ("router", question_router, {"question": "Who won the cricket world cup?"}, "websearch"),
    ("retrieval", retrieval_grader,
     {"question": Q, "document": "Agents keep short-term and long-term memory."}, True),
    ("retrieval", retrieval_grader,
     {"question": Q, "document": "Pizza dough needs flour and yeast."}, False),
    ("hallucination", hallucination_grader, {"documents": DOCS, "generation": DOCS}, True),
    ("hallucination", hallucination_grader,
     {"documents": DOCS, "generation": "Agents store memory in quantum crystals."}, False),
    ("answer", answer_grader,
     {"question": Q, "generation": "Agent memory lets an agent keep information."}, True),
    ("answer", answer_grader, {"question": Q, "generation": "I don't know."}, False),
]

correct = 0
for label, chain, inputs, expected in CASES:
    result = chain.invoke(inputs)
    got = getattr(result, "datasource", None) or result.binary_score
    ok = got == expected
    correct += ok
    print(f"{label:14} expected={str(expected):12} got={str(got):12} {'ok' if ok else 'MISS'}")

print(f"\n{correct}/{len(CASES)} matched")