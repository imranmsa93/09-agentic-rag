from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

tavily_search = TavilySearch(max_results =3)

@tool
def search_web(query:str)-> Document:
    response  = tavily_search.invoke({"query":query})
    parts = [ f'source:{r.get("url")}\nContent:{r.get("content")}'
                for r in response["results"]
                ]
    docs = Document(page_content="\n\n".join(parts),metaData={"source":"web"})
    return docs


