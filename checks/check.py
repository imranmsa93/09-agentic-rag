from langchain_core.tools import tool

try:
    @tool
    def no_doc(query: str) -> str:
        return query
except Exception as e:
    print("1:", type(e).__name__)

@tool
def with_doc(query: str) -> str:
    """Echo the query."""
    return query

@tool(description="Echo the query back.")
def with_description(query: str) -> str:
    return query

print("2:", with_doc.description)
print("3:", with_description.description)