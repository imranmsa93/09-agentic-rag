import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import run_question
from rag.core.config import NO_ANSWER

QUESTIONS = [
    "What is agent memory?",
    "Who won the cricket world cup?",
    "What is chain of thought prompting?",
    "What did the authors say about quantum computing?",
]

for q in QUESTIONS:
    path, state = run_question(q)
    print(f"\nQ: {q}")
    print("   path    :", " -> ".join(path))
    print("   attempts:", state.get("attempts"), "| refusal:", NO_ANSWER in state.get("generation", ""))