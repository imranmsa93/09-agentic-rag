from rag.chains.prompts import (
    answer_grader_prompt, hallucination_grader_prompt,
    retrieval_grader_prompt, router_prompt,
)
from rag.chains.schemas import (
    GradeAnswer, GradeDocuments, GradeHallucinations, RouteQuery,
)
from rag.core.llm import llm
from rag.data.search import search_web

from langchain_core.output_parsers import StrOutputParser
from rag.chains.prompts import generation_prompt   # add to the existing prompts import

search_tool = llm.bind_tools([search_web])
generation_chain = generation_prompt | llm | StrOutputParser()

question_router = router_prompt | llm.with_structured_output(RouteQuery)
retrieval_grader = retrieval_grader_prompt | llm.with_structured_output(GradeDocuments)
hallucination_grader = hallucination_grader_prompt | llm.with_structured_output(GradeHallucinations)
answer_grader = answer_grader_prompt | llm.with_structured_output(GradeAnswer)