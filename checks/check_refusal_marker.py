import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag.core.llm import llm

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer the question using only the context, in at most three sentences. "
     "Do not add facts that are not in the context. "
     "If the context does not contain the answer, reply with exactly NO_ANSWER "
     "and nothing else."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])
chain = prompt | llm | StrOutputParser()

CONTEXT = (
    "Agent memory has two parts. Short-term memory is the model's in-context "
    "learning. Long-term memory is an external store the agent can query."
)

CASES = [
    ("What are the two parts of agent memory?", False),
    ("What is long-term memory?", False),
    ("Who won the cricket world cup?", True),
    ("What is the capital of France?", True),
]

for q, should_refuse in CASES:
    a = chain.invoke({"context": CONTEXT, "question": q}).strip()
    exact = a == "NO_ANSWER"
    status = "ok" if exact == should_refuse else "MISS"
    print(f"{status:4} refuse_expected={str(should_refuse):5} exact={str(exact):5} "
          f"contains={'NO_ANSWER' in a}\n     Q: {q}\n     A: {a}\n")