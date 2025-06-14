import os
from rich.console import Console
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv(override=True)
console = Console()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily_client = None

if TAVILY_API_KEY:
    try:
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        console.log("Tavily client initialized successfully.", style="bold green")
    except Exception as e:
        console.log(f"Failed to initialize Tavily client: {e}", style="bold red")
else:
    console.log("TAVILY_API_KEY not found in environment variables.", style="bold red")


def search_tavily(
    query: str,
    max_res: int = 5,
    **kwargs,
    # search_depth: str = "advanced",
) -> dict:
    """
    Search using Tavily API with a simple query.
    Args:
        query (str): The search query
        max_results (int): Maximum number of results to return (default: 5)
        **kwargs: Additional Tavily search parameters

    Returns:
        dict: Search results from Tavily API
    """

    if not tavily_client:
        return {
            "error": "Tavily client not initialized. Check your API key.",
            "results": [],
        }
    if not query or not query.strip():
        return {"error": "query cannot be empty", "results": []}
    try:
        console.print(f"🔍 Searching for: '{query}'", style="bold blue")
        results = tavily_client.search(
            query=query,
            **kwargs,
        )

        if results and results.get("results"):
            console.print(f"✅ Found {len(results['results'])} results", style="green")
            return results
        else:
            console.print("⚠️ No results found", style="yellow")
            return {"error": "No results found for your query", "results": []}

    except Exception as e:
        error_msg = f"Search failed: {str(e)}"
        console.print(f"❌ {error_msg}", style="bold red")
        return {"error": error_msg, "results": []}


if __name__ == "__main__":
    # Test the search function
    test_query = input("Enter your search query: ").strip()

    if test_query:
        # Perform search
        search_results = search_tavily(test_query, max_results=3)
        print(search_results)
