import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.chains.schemas import (
    GradeAnswer, GradeDocuments, GradeHallucinations, RouteQuery,
)
from rag.core.llm import llm

CASES = [
    (RouteQuery, "What is agent memory?", "vectorstore"),
    (RouteQuery, "Who won the cricket world cup?", "websearch"),
    (GradeDocuments,
     "Question: What is agent memory?\nDocument: Agents keep short-term and long-term memory.",
     True),
    (GradeDocuments,
     "Question: What is agent memory?\nDocument: Pizza dough needs flour and yeast.",
     False),
    (GradeHallucinations,
     "Documents: Agent memory has short-term and long-term parts.\n"
     "Answer: Agent memory has short-term and long-term parts.",
     True),
    (GradeHallucinations,
     "Documents: Agent memory has short-term and long-term parts.\n"
     "Answer: Agents store memory in quantum crystals.",
     False),
    (GradeAnswer,
     "Question: What is agent memory?\nAnswer: Agent memory lets an agent keep information.",
     True),
    (GradeAnswer,
     "Question: What is agent memory?\nAnswer: I don't know.",
     False),
]

for schema, text, expected in CASES:
    result = llm.with_structured_output(schema).invoke(text)
    got = getattr(result, "datasource", None) or getattr(result, "binary_score", None)
    print(f"{schema.__name__:20} expected={str(expected):12} got={got}")