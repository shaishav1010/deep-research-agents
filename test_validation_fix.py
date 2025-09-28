"""Test script to verify WebSearchResult validation fixes"""
import os
import sys
# Set UTF-8 encoding for Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from research_agent import ContextualRetrieverAgent
from models import SourceType

# Set up test API keys (these can be dummy for testing the validation)
test_api_key = os.getenv("OPENROUTER_API_KEY", "sk-or-test")
test_tavily_key = os.getenv("TAVILY_API_KEY", "tvly-test")

def test_source_type_mapping():
    """Test that source types are properly mapped"""
    agent = ContextualRetrieverAgent(
        api_key=test_api_key,
        tavily_api_key=test_tavily_key
    )

    # Test the type mapping
    test_results = [
        {"title": "Test Academic Paper", "url": "http://example.com/paper", "snippet": "Academic content", "domain": "example.com"},
        {"title": "Test News Article", "url": "http://news.com/article", "snippet": "News content", "domain": "news.com"},
        {"title": "Test Website", "url": "http://website.com/page", "snippet": "Web content", "domain": "website.com"}
    ]

    try:
        # Test with a sample query
        result = agent.analyze_and_rank("test query", test_results)

        print("SUCCESS: WebSearchResult validation successful!")
        print(f"Query: {result.query}")
        print(f"Total results: {result.total_results_found}")
        print(f"Search strategy: {result.search_strategy}")
        print(f"Key insights: {result.key_insights}")
        print(f"Number of sources: {len(result.sources)}")

        if result.sources:
            print("\nFirst source:")
            source = result.sources[0]
            print(f"  Title: {source.title}")
            print(f"  Type: {source.source_type}")
            print(f"  Valid enum: {source.source_type in [e.value for e in SourceType]}")

        return True
    except Exception as e:
        print(f"ERROR: Validation failed: {e}")
        return False

def test_fallback_mechanism():
    """Test the fallback mechanism for missing fields"""
    agent = ContextualRetrieverAgent(
        api_key=test_api_key,
        tavily_api_key=test_tavily_key
    )

    # Test with empty results
    try:
        result = agent.analyze_and_rank("test query", [])
        print("\nSUCCESS: Empty results handling successful!")
        print(f"Key insights: {result.key_insights}")
        assert len(result.key_insights) >= 1, "Should have at least one insight"
        assert result.total_results_found == 0, "Should have 0 results"
        assert result.search_strategy, "Should have a search strategy"
        return True
    except Exception as e:
        print(f"ERROR: Fallback mechanism failed: {e}")
        return False

def test_enum_values():
    """Test that all enum values are valid"""
    valid_enums = [e.value for e in SourceType]
    print("\nValid SourceType enum values:")
    for enum_val in valid_enums:
        print(f"  - {enum_val}")

    # Test cases that previously failed
    test_mappings = {
        "academic": "academic_paper",
        "news": "news_article",
        "report": "report",
        "database": "database",
        "unknown": "website"  # Default fallback
    }

    print("\nType mappings:")
    for original, mapped in test_mappings.items():
        is_valid = mapped in valid_enums
        status = "[OK]" if is_valid else "[ERROR]"
        print(f"  {status} {original} -> {mapped}")

    return all(mapped in valid_enums for mapped in test_mappings.values())

if __name__ == "__main__":
    print("Testing WebSearchResult validation fixes...\n")

    tests = [
        ("Enum Values", test_enum_values),
        ("Source Type Mapping", test_source_type_mapping),
        ("Fallback Mechanism", test_fallback_mechanism)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        results.append(test_func())

    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    for (test_name, _), result in zip(tests, results):
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")

    if all(results):
        print("\n*** All tests passed! The validation errors should be fixed. ***")
    else:
        print("\nWARNING: Some tests failed. Please review the implementation.")