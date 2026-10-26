import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from rag.core.llm import llm


class A(BaseModel):                     # current: vague name
    binary_score: bool = Field(description="True if the document is relevant to the question, otherwise False.")


class B(BaseModel):                     # meaningful name
    is_relevant: bool = Field(description="True if the document helps answer the question.")


class C(BaseModel):                     # reason first, then verdict
    reason: str = Field(description="One short sentence explaining the verdict.")
    is_relevant: bool = Field(description="True if the document helps answer the question.")


prompt = ChatPromptTemplate.from_messages([
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

for name, schema in [("A binary_score", A), ("B is_relevant", B), ("C reason + is_relevant", C)]:
    chain = prompt | llm.with_structured_output(schema)
    correct = 0
    print(f"\n--- variant {name}")
    for doc, expected in CASES:
        r = chain.invoke({"question": Q, "document": doc})
        got = getattr(r, "binary_score", None)
        if got is None:
            got = r.is_relevant
        correct += got == expected
        print(f"  expected={str(expected):5} got={str(got):5}  {doc[:42]}")
    print(f"  {correct}/{len(CASES)}")