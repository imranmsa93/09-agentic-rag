import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
import asyncio


urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
embeddings = OllamaEmbeddings(model="embeddinggemma")
vector_store = PineconeVectorStore(index_name=os.environ["INDEX_NAME"] ,embedding=embeddings)

retriever = vector_store.as_retriever(search_kwargs = {"k":5})


def scrapper():
    docs = [WebBaseLoader(url).load() for url in urls]
    doclist = [doc for item in docs for doc in item]
    print("Working with RecursiveCharacterTextSplitter for text Splitting")
    splitter = RecursiveCharacterTextSplitter(
            chunk_size=250,
            chunk_overlap=25,
        
        )
    chunks = splitter.split_documents(doclist)
    print(f"Original documents: {len(doclist)}")
    print(f"Chunks created: {len(chunks)}")

    return chunks

def do_embeddings(chunks):
    
    print("Connected to Pinecone index")
    vector_store.add_documents(chunks)
    print(f"{len(chunks)} documents added to Pinecone")




async def main():
    print("Running from main")
    chunks =  scrapper()
    do_embeddings(chunks)

if __name__ == "__main__":
    asyncio.run(main())