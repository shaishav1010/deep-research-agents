import os
from research_agent import ContextualRetrieverAgent

# Test just the search functionality without LLM
def test_search_only():
    print("Testing search functionality...")

    # Create search node with dummy keys
    search_node = ContextualRetrieverAgent(
        api_key="sk-or-dummy",  # Won't be used for search
        tavily_api_key="tvly-test"  # Dummy key for testing
    )

    # Test query
    query = "Major EV charging infrastructure players"
    print(f"\nSearching for: {query}")

    # Test the search function directly
    search_results = search_node.search_web(query)

    print(f"\nFound {len(search_results)} search results")

    if search_results:
        for i, result in enumerate(search_results[:3], 1):
            print(f"\n{i}. {result['title']}")
            print(f"   URL: {result['url']}")
            print(f"   Domain: {result['domain']}")
            print(f"   Snippet: {result['snippet'][:100]}...")
    else:
        print("\nNo results returned from search_web")

    # Now test the full flow with fallback ranking
    print("\n\nTesting full analysis with fallback ranking...")
    web_result = search_node.analyze_and_rank(query, search_results)

    print(f"\nWeb result sources: {len(web_result.sources)}")
    print(f"Total found: {web_result.total_results_found}")

    for i, source in enumerate(web_result.sources[:3], 1):
        print(f"\n{i}. {source.title}")
        print(f"   Relevance: {source.relevance_score}%")
        print(f"   Type: {source.source_type}")

if __name__ == "__main__":
    test_search_only()