import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.data.vectorstore import retriever

docs = retriever.invoke("agent memory")
print("chunks returned:", len(docs))
for d in docs:
    text = d.page_content.replace("\n", " ")
    print(f"{len(d.page_content):4} chars | {d.metadata.get('source')} | {text[:90]}")