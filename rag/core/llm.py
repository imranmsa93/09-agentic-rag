from langchain_ollama import ChatOllama

MODEL = "qwen3:1.7b"

llm = ChatOllama(model=MODEL, temperature=0)