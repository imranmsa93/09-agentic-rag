from rag.nodes import GraphState
from typing import Any, Dict
from ingestion import retreiver
from ingestion import vectorStore

def find_query(state:GraphState) -> Dict[str,Any]:
    print("---RETRIEVER CALLED---")
    question = state["question"]
    documents = retreiver.invoke(question)
    print("---RETRIEVED--->",documents)
    return {"documents": documents, "question": question}

find_query("Autonomous Agents")