import os
from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

embeddings = OllamaEmbeddings(model="embeddinggemma")
vector_store = PineconeVectorStore(index_name=os.environ["INDEX_NAME"] ,embedding=embeddings)

retriever = vector_store.as_retriever(search_kwargs = {"k":5})

