import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from rag.chains.prompts import retrieval_grader_prompt
from rag.chains.schemas import GradeDocuments
from rag.core.llm import llm


class NoDoc(BaseModel):                 # same field as GradeDocuments, no docstring
    binary_score: bool = Field(
        description="True if the document is relevant to the question, otherwise False."
    )


variant_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You grade whether a document is relevant to a question. Relevant means the "
     "document contains information that helps answer the question."),
    ("human", "Document:\n{document}\n\nQuestion: {question}"),
])

Q = "What is agent memory?"
CASES = [
    ("Agents keep short-term and long-term memory.", True),
    ("Memory lets an agent store and recall past information.", True),
    ("Pizza dough needs flour and yeast.", False),
    ("The Eiffel Tower is in Paris.", False),
]

COMBOS = [
    ("1 real prompt  + real schema (docstring)", retrieval_grader_prompt, GradeDocuments),
    ("2 real prompt  + no docstring", retrieval_grader_prompt, NoDoc),
    ("3 new prompt   + real schema (docstring)", variant_prompt, GradeDocuments),
    ("4 new prompt   + no docstring", variant_prompt, NoDoc),
]

for name, prompt, schema in COMBOS:
    chain = prompt | llm.with_structured_output(schema)
    correct = sum(
        chain.invoke({"question": Q, "document": d}).binary_score == exp
        for d, exp in CASES
    )
    print(f"{name:45} {correct}/{len(CASES)}")