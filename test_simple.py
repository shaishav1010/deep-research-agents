"""Simple test to check data flow"""
from research_agent import ResearchGraph

# Create research graph with test keys
graph = ResearchGraph(
    api_key="sk-or-test",  # Dummy key
    tavily_api_key="tvly-test"  # Dummy key
)

# Test the flow
query = "Impact of mobile phones on kids brain"
print(f"Testing query: {query}\n")

try:
    # Run the initial state to test data flow
    initial_state = {
        "research_query": query,
        "search_results": None,
        "critical_analysis": None,
        "insight_analysis": None,
        "current_step": "initial",
        "error": None
    }

    # Just test the retriever agent
    retriever = graph.retriever_agent
    result = retriever.execute(initial_state)

    print(f"Retriever result:")
    print(f"- Step: {result.get('current_step')}")
    print(f"- Error: {result.get('error')}")
    if result.get('search_results'):
        sr = result['search_results']
        print(f"- Sources found: {len(sr.sources)}")
        print(f"- Total results: {sr.total_results_found}")
    else:
        print("- No search results")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()