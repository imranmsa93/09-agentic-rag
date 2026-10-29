from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

tavily_search = TavilySearch(max_results =3)

@tool
def search_web(query:str)-> Document:
    """Search the web and return the results as one Document.

    Use for questions the local documents cannot answer. Each result in the
    text starts with its source URL.
    """
    response  = tavily_search.invoke({"query":query})
    parts = [ f'source:{r.get("url")}\nContent:{r.get("content","")}'
                for r in response["results"]
                if r.get("url")
                ]
    docs = Document(page_content="\n\n".join(parts),metadata={"source":"web"})
    return docs


