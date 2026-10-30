import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.data.vectorstore import vector_store

for q in ["agent memory", "prompt engineering", "adversarial attacks", "diffusion models"]:
    docs = vector_store.similarity_search(q, k=10)
    sources = sorted({d.metadata.get("source", "?") for d in docs})
    print(f"\n{q}")
    for s in sources:
        print("  ", s)