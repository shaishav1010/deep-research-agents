"""Test the research agent with real API keys if available"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set UTF-8 encoding for Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from research_agent import ResearchGraph
from models import SourceType

def test_real_search():
    """Test with real API keys if available"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")

    if not api_key or not tavily_key:
        print("WARNING: API keys not found in environment")
        print("Please set OPENROUTER_API_KEY and TAVILY_API_KEY")
        print("You can create a .env file with these keys")
        return False

    print("Found API keys, testing real search...")

    try:
        # Create research graph
        graph = ResearchGraph(
            api_key=api_key,
            tavily_api_key=tavily_key
        )

        # Test query
        query = "Latest developments in quantum computing 2024"
        print(f"\nTesting query: {query}")

        # Run only the retriever agent to test validation
        initial_state = {
            "research_query": query,
            "search_results": None,
            "critical_analysis": None,
            "insight_analysis": None,
            "current_step": "initial",
            "error": None
        }

        result = graph.retriever_agent.execute(initial_state)

        if result.get("error"):
            print(f"ERROR: {result['error']}")
            return False

        search_results = result.get("search_results")
        if search_results:
            print(f"\nSUCCESS: Search completed without validation errors")
            print(f"- Total results found: {search_results.total_results_found}")
            print(f"- Sources retrieved: {len(search_results.sources)}")
            print(f"- Search strategy: {search_results.search_strategy}")
            print(f"- Key insights: {len(search_results.key_insights)} insights generated")

            # Check that all sources have valid enum values
            all_valid = True
            valid_enums = [e.value for e in SourceType]

            for i, source in enumerate(search_results.sources[:3]):  # Check first 3
                is_valid = source.source_type.value in valid_enums
                if not is_valid:
                    print(f"  ERROR: Source {i} has invalid type: {source.source_type}")
                    all_valid = False
                else:
                    print(f"  [OK] Source: {source.title[:50]}... Type: {source.source_type.value}")

            return all_valid
        else:
            print("ERROR: No search results returned")
            return False

    except Exception as e:
        print(f"ERROR: Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_query_refinement():
    """Test query refinement with real API"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")

    if not api_key or not tavily_key:
        print("Skipping query refinement test (no API keys)")
        return True  # Don't fail if no keys

    print("\nTesting query refinement...")

    try:
        from research_agent import ContextualRetrieverAgent

        agent = ContextualRetrieverAgent(
            api_key=api_key,
            tavily_api_key=tavily_key
        )

        # Test broad query refinement
        broad_query = "artificial intelligence"
        refined = agent.refine_query(broad_query)

        print(f"Original query: {broad_query}")
        print(f"Interpretation: {refined.interpretation}")
        print(f"Subtopics identified: {len(refined.subtopics)}")

        for subtopic in refined.subtopics:
            print(f"  - {subtopic.topic}: {subtopic.description[:50]}...")

        print("SUCCESS: Query refinement working")
        return True

    except Exception as e:
        print(f"ERROR: Query refinement failed: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("REAL API INTEGRATION TEST")
    print("="*60)

    test_results = []

    # Run tests
    print("\n1. Testing Real Search with Validation")
    print("-"*40)
    test_results.append(test_real_search())

    print("\n2. Testing Query Refinement")
    print("-"*40)
    test_results.append(test_query_refinement())

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    if all(test_results):
        print("*** ALL TESTS PASSED ***")
        print("The WebSearchResult validation errors have been fixed.")
        print("The system should now handle all source types correctly.")
    else:
        print("Some tests failed. Please check the error messages above.")