import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.chains.chains import retrieval_grader
from rag.data.vectorstore import retriever

QUESTIONS = [
    "What are the two parts of agent memory?",
    "What types of memory does an LLM agent have?",
]

for q in QUESTIONS:
    print(f"\n##### {q}")
    for i, d in enumerate(retriever.invoke(q)):
        verdict = retrieval_grader.invoke(
            {"question": q, "document": d.page_content}
        ).binary_score
        text = d.page_content.replace("\n", " ")
        src = d.metadata.get("source", "")[-32:]
        print(f"[{i}] relevant={verdict} | {len(d.page_content)} chars | {src} | {text[:110]}")