import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.nodes import (
    generate_node, grade_documents_node, retriever_node, web_search_node,
)

state = {"question": "What are the two parts of agent memory?"}


def run(label, node):
    print(f"\n=== Before {label} Keys: {list(state.keys())}") 
    update = node(state)
    print(f"\n=== {label} wrote: {list(update.keys())}")
    state.update(update)


run("retrieve", retriever_node)
print("chunks fetched:", len(state["documents"]))

run("grade_documents", grade_documents_node)
print("chunks kept:", len(state["documents"]), "| web_search flag:", state["web_search"])

run("generate", generate_node)
print("ANSWER:", state["generation"])
print("attempts:", state["attempts"])

# The web node on its own, with no prior documents
state2 = {"question": "Who won the cricket world cup?"}
state2.update(web_search_node(state2))
print("\nweb documents:", len(state2["documents"]))
print(state2["documents"][0].page_content[:200])