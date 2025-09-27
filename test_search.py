from duckduckgo_search import DDGS
import asyncio

def test_sync_search():
    print("Testing synchronous search...")
    with DDGS() as ddgs:
        query = "artificial intelligence"
        results = list(ddgs.text(query, max_results=5))
        print(f"Found {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.get('title', 'No title')}")
            print(f"   URL: {result.get('link', result.get('href', 'No URL'))}")
            print(f"   Snippet: {result.get('body', 'No snippet')[:100]}...")

if __name__ == "__main__":
    test_sync_search()