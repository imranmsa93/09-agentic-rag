from typing import Literal

from pydantic import BaseModel, Field


class RouteQuery(BaseModel):
    """Route a user question to the most relevant data source."""

    datasource: Literal["vectorstore", "websearch"] = Field(
        description=(
            "Use 'vectorstore' for questions about LLM agents, prompt engineering "
            "and adversarial attacks on LLMs. Use 'websearch' for anything else."
        )
    )


class GradeDocuments(BaseModel):
    """Is a retrieved document relevant to the question?"""

    binary_score: bool = Field(
        description="True if the document is relevant to the question, otherwise False."
    )


class GradeHallucinations(BaseModel):
    """Is the answer grounded in the documents?"""

    binary_score: bool = Field(
        description="True if the answer is supported by the documents, otherwise False."
    )


class GradeAnswer(BaseModel):
    """Does the answer address the question?"""

    binary_score: bool = Field(
        description="True if the answer addresses the question, otherwise False."
    )