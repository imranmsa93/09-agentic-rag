import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.graph import graph

print(graph.get_graph().draw_mermaid())

question = "Who won the cricket world cup?"
for step in graph.stream(
    {"question": question},
    stream_mode="updates",
    config={"recursion_limit": 15},
):
    for node, update in step.items():
        print(f"-> {node:16} wrote {list(update.keys())}")