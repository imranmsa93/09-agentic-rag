from langgraph.graph import END, START, StateGraph

from rag.core.config import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from rag.core.state import GraphState
from rag.nodes import (
    generate_node, grade_documents_node, retriever_node, web_search_node,
)
from rag.routing import (
    decide_after_generation, decide_after_grading, route_question,
)

builder = StateGraph(GraphState)

builder.add_node(RETRIEVE, retriever_node)
builder.add_node(GRADE_DOCUMENTS, grade_documents_node)
builder.add_node(WEBSEARCH, web_search_node)
builder.add_node(GENERATE, generate_node)

builder.add_conditional_edges(
    START, route_question, {RETRIEVE: RETRIEVE, WEBSEARCH: WEBSEARCH}
)
builder.add_edge(RETRIEVE, GRADE_DOCUMENTS)
builder.add_conditional_edges(
    GRADE_DOCUMENTS, decide_after_grading, {WEBSEARCH: WEBSEARCH, GENERATE: GENERATE}
)
builder.add_edge(WEBSEARCH, GENERATE)
builder.add_conditional_edges(
    GENERATE, decide_after_generation,
    {END: END, WEBSEARCH: WEBSEARCH, GENERATE: GENERATE},
)

graph = builder.compile()