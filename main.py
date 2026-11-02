
import sys

from rag.graph import graph

DEFAULT_QUESTION = "What is agent memory?"


def main():
    print("09-agentic-rag - Implementation of Advanced RAG!")
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION

    state = {"question": question}
    path = []
    for step in graph.stream(
        {"question": question},
        stream_mode="updates",
        config={"recursion_limit": 15},
    ):
       #print("STEP:", step)
        for node, update in step.items():
            path.append(node)
            state.update(update)

    print("QUESTION:", question)
    print("PATH    :", " -> ".join(path))
    print("\nANSWER:\n", state.get("generation"))
    print("\nSOURCES:")
    for d in state.get("documents", []):
        print(" -", d.metadata.get("source"))
    print("\nATTEMPTS:", state.get("attempts"))


if __name__ == "__main__":
    main()