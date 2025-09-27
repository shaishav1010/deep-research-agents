"""Test visualization generation directly"""
import sys
from models import (
    ResearchState, WebSearchResult, ResearchSource, SourceType,
    CriticalAnalysis, KeyFinding, InsightAnalysis, VisualizationSpec
)
from research_agent import InsightGenerationAgent

# Create mock data
def create_mock_data():
    # Create mock search results
    sources = [
        ResearchSource(
            title="Study on mobile phone impact",
            url="http://example.com",
            snippet="Mobile phones affect children's brain development",
            source_type=SourceType.ACADEMIC_PAPER,
            relevance_score=90,
            domain="example.com",
            reasoning="Relevant to query"
        )
    ]

    search_results = WebSearchResult(
        query="Impact of mobile phones on kids brain",
        sources=sources,
        total_results_found=10,
        search_strategy="Multi-source search",
        key_insights=["Mobile phones impact brain development"]
    )

    # Create mock analysis
    critical_analysis = CriticalAnalysis(
        executive_summary="Mobile phones have significant impact on children's brain development",
        key_findings=[
            KeyFinding(
                finding="Screen time affects cognitive development",
                sources=["Study 1"],
                confidence=0.8
            )
        ],
        contradictions=[],
        source_validations=[],
        consensus_points=["Extended screen time affects attention span"],
        gaps_identified=["More research needed on long-term effects"],
        recommendations=["Limit screen time for young children"],
        confidence_assessment="High confidence based on multiple studies"
    )

    return search_results, critical_analysis

# Test the insight generation
def test_insight_generation():
    print("Testing insight generation...")

    search_results, critical_analysis = create_mock_data()

    # Create agent (with dummy API key since we'll use fallback)
    agent = InsightGenerationAgent(api_key="dummy")

    # Generate insights
    query = "Impact of mobile phones on kids brain. Summarize between age groups."
    insights = agent._generate_fallback_insights(query, search_results, critical_analysis)

    print(f"\nGenerated {len(insights.visualizations)} visualizations:")
    for i, viz in enumerate(insights.visualizations, 1):
        print(f"\n{i}. {viz.title}")
        print(f"   Type: {viz.chart_type}")
        if viz.chart_type == "pie":
            print(f"   Labels: {viz.data.get('labels', [])}")
            print(f"   Values: {viz.data.get('values', [])}")
        else:
            print(f"   X data: {viz.data.get('x', [])}")
            print(f"   Y data: {viz.data.get('y', [])}")

    print(f"\nKey Statistics:")
    for key, value in insights.key_statistics.items():
        print(f"   {key}: {value}")

    print(f"\nTest passed! Visualizations generated successfully.")
    return insights

if __name__ == "__main__":
    insights = test_insight_generation()