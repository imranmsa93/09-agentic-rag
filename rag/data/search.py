from graph.state import GraphState
from typing import Any, Dict
from ingestion import retreiver
from ingestion import vectorStore

def find_query(state:GraphState) -> Dict[str,Any]:
    print("---RETRIEVE---")
    question = state["question"]
    documents = retreiver.invoke(question)
    print(documents)
    return {"documents": documents, "question": question}

find_query("Autonomous Agents")