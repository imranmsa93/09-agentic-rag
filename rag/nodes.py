import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.core.state import GraphState
from rag.data.vectorstore import retriever
from rag.data.search import search_web
from rag.chains.chains import retrieval_grader, generation_chain

def retriever_node(state:GraphState):
    doc = retriever.invoke(state["question"])
    document = {"documents":doc}
    return document

def web_search_node(state:GraphState):
    response  = search_web.invoke({"query": state["question"] })
    documents =  state.get("documents",[]) + [response]
    return {"documents":documents}

def grade_documents_node(state: GraphState):
    kept = []
    web_search = False
    for doc in state["documents"]:
        verdict = retrieval_grader.invoke({"document":doc.page_content, "question":state["question"]})
        if verdict.binary_score:
             kept.append(doc)
        else:
            web_search = True
    return {"documents":kept, "web_search": web_search}
        

def generate_node(state: GraphState):
    context = "\n\n".join(d.page_content for d in state.get("documents", []))
    generation = generation_chain.invoke(
        {"context": context, "question": state["question"]}
    )
    return {"generation": generation, "attempts": state.get("attempts", 0) + 1}

