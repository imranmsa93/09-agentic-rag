from typing import List, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict, total=False):
    """
        Represents the state of our graph.
        Attributes:
            question: question
            generation: LLM generation
            web_search: whether to add search
            documents: list of documents
    """
    
    question: str
    documents: List[Document]
    generation: str
    web_search: bool     # set by grade_documents: True if any document was dropped
    attempts: int        # number of generate calls, to cap the retry loop