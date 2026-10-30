from langgraph.graph import END

from rag.chains.chains import answer_grader, hallucination_grader, question_router
from rag.core.config import GENERATE, MAX_ATTEMPTS, NO_ANSWER, RETRIEVE, WEBSEARCH
from rag.core.state import GraphState


def route_question(state: GraphState) -> str:
    source = question_router.invoke({"question": state["question"]})
    return WEBSEARCH if source.datasource == "websearch" else RETRIEVE


def decide_after_grading(state: GraphState) -> str:
    return WEBSEARCH if state.get("web_search", False) else GENERATE


def decide_after_generation(state: GraphState) -> str:
    generation = state["generation"]
    out_of_attempts = state["attempts"] >= MAX_ATTEMPTS

    if NO_ANSWER in generation:                    # the model found nothing in the context
        return END if out_of_attempts else WEBSEARCH

    context = "\n\n".join(d.page_content for d in state.get("documents", []))
    grounded = hallucination_grader.invoke(
        {"documents": context, "generation": generation}
    )
    if not grounded.binary_score:
        return END if out_of_attempts else GENERATE

    useful = answer_grader.invoke(
        {"question": state["question"], "generation": generation}
    )
    if useful.binary_score:
        return END
    return END if out_of_attempts else WEBSEARCH