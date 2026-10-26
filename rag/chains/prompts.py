from langchain_core.prompts import ChatPromptTemplate
from rag.core.config import NO_ANSWER

router_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You route user questions. Use 'vectorstore' if the question is about LLM agents "
     "(planning, memory, tools), prompt engineering, or adversarial attacks on LLMs. "
     "Use 'websearch' for everything else."),
    ("human", "{question}"),
])

retrieval_grader_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You grade whether a document is relevant to a question. Relevant means the "
     "document contains information that helps answer the question."),
    ("human", "Document:\n{document}\n\nQuestion: {question}"),
])

hallucination_grader_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You check whether an answer is supported by a set of documents. "
     "Answer True only if every claim in the answer appears in the documents. "
     "If any claim is not in the documents, answer False."),
    ("human", "Documents:\n{documents}\n\nAnswer: {generation}"),
])

answer_grader_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You grade whether an answer addresses a question. An answer addresses the "
     "question when it gives information that responds to it. 'I don't know', "
     "refusals and off-topic replies do not address the question."),
    ("human", "Question: {question}\n\nAnswer: {generation}"),
])

generation_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer the question using only the context, in at most three sentences. "
     "Do not add facts that are not in the context. "
     f"If the context does not contain the answer, reply with exactly {NO_ANSWER} "
     "and nothing else."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])