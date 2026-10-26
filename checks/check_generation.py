import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.chains.chains import answer_grader, generation_chain, hallucination_grader
from rag.core.config import NO_ANSWER
CONTEXT = (
    "Agent memory has two parts. Short-term memory is the model's in-context "
    "learning. Long-term memory is an external store the agent can query."
)

QUESTIONS = [
    "What are the two parts of agent memory?",
    "What is long-term memory?",
    "Who won the cricket world cup?",
]

for q in QUESTIONS:
    answer = generation_chain.invoke({"context": CONTEXT, "question": q})
    print(f"\nQ: {q}\nA: {answer}")
    if NO_ANSWER in answer:
        print("   -> refusal marker found: send to web search (judges skipped)")
    else:
        grounded = hallucination_grader.invoke({"documents": CONTEXT, "generation": answer})
        useful = answer_grader.invoke({"question": q, "generation": answer})
        print(f"   grounded={grounded.binary_score}   addresses_question={useful.binary_score}")