from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class SourceType(str, Enum):
    ACADEMIC_PAPER = "academic_paper"
    NEWS_ARTICLE = "news_article"
    BLOG_POST = "blog_post"
    DOCUMENTATION = "documentation"
    FORUM = "forum"
    VIDEO = "video"
    REPORT = "report"
    BOOK = "book"
    WEBSITE = "website"
    DATABASE = "database"
    OTHER = "other"


class ResearchSource(BaseModel):
    title: str = Field(description="Title of the source")
    url: str = Field(description="URL of the source")
    snippet: str = Field(description="Brief excerpt or summary from the source")
    source_type: SourceType = Field(description="Type of source")
    relevance_score: float = Field(
        description="Relevance score from 0-100", ge=0, le=100
    )
    published_date: Optional[str] = Field(
        default=None, description="Publication date if available"
    )
    author: Optional[str] = Field(default=None, description="Author(s) if available")
    domain: str = Field(description="Domain/website of the source")
    reasoning: str = Field(
        description="Explanation of why this source is relevant to the research query"
    )


class Subtopic(BaseModel):
    topic: str = Field(description="Subtopic title")
    description: str = Field(description="Brief description of what this subtopic covers")
    search_queries: List[str] = Field(description="Refined search queries for this subtopic")
    importance: float = Field(description="Importance score (0-1)", ge=0, le=1)

class RefinedQuery(BaseModel):
    original_query: str = Field(description="Original user query")
    interpretation: str = Field(description="How the query was interpreted")
    refined_queries: List[str] = Field(description="List of refined, more specific queries")
    subtopics: List[Subtopic] = Field(description="Breakdown into subtopics")
    search_strategy: str = Field(description="Strategy for searching diverse sources")

class WebSearchResult(BaseModel):
    query: str = Field(description="Original research query")
    refined_query: Optional[RefinedQuery] = Field(
        default=None, description="Refined and interpreted query with subtopics"
    )
    search_timestamp: datetime = Field(
        default_factory=datetime.now, description="When the search was performed"
    )
    sources: List[ResearchSource] = Field(
        description="List of top sources ranked by relevance"
    )
    sources_by_subtopic: Optional[Dict[str, List[ResearchSource]]] = Field(
        default=None, description="Sources organized by subtopic"
    )
    total_results_found: int = Field(
        description="Total number of results found before filtering"
    )
    search_strategy: str = Field(
        description="Description of the search strategy used"
    )
    key_insights: List[str] = Field(
        description="Key insights or patterns observed across sources"
    )


class Contradiction(BaseModel):
    source1: str = Field(description="Title or reference of first source")
    source2: str = Field(description="Title or reference of second source")
    claim1: str = Field(description="Claim from first source")
    claim2: str = Field(description="Conflicting claim from second source")
    explanation: str = Field(description="Explanation of why these claims contradict")

class KeyFinding(BaseModel):
    finding: str = Field(description="Key finding or insight")
    sources: List[str] = Field(description="Sources supporting this finding")
    confidence: float = Field(description="Confidence level (0-1)", ge=0, le=1)

class SourceValidation(BaseModel):
    source_title: str = Field(description="Title of the source")
    credibility_score: float = Field(description="Credibility score (0-100)", ge=0, le=100)
    credibility_factors: List[str] = Field(description="Factors affecting credibility")
    potential_biases: List[str] = Field(description="Identified potential biases")

class CriticalAnalysis(BaseModel):
    executive_summary: str = Field(description="High-level executive summary of findings")
    key_findings: List[KeyFinding] = Field(description="List of key findings from all sources")
    contradictions: List[Contradiction] = Field(description="Identified contradictions between sources")
    source_validations: List[SourceValidation] = Field(description="Validation of top sources")
    consensus_points: List[str] = Field(description="Points where multiple sources agree")
    gaps_identified: List[str] = Field(description="Knowledge gaps or areas needing more research")
    recommendations: List[str] = Field(description="Recommended next steps or actions")
    confidence_assessment: str = Field(description="Overall confidence assessment of the research")
    analysis_timestamp: datetime = Field(default_factory=datetime.now)

class TrendData(BaseModel):
    label: str = Field(description="Trend label or time period")
    value: float = Field(description="Trend value")
    category: str = Field(description="Category this trend belongs to")

class CategoryDistribution(BaseModel):
    category: str = Field(description="Category name")
    count: int = Field(description="Number of items in category")
    percentage: float = Field(description="Percentage of total")
    description: str = Field(description="Description of what this category represents")

class StatisticalInsight(BaseModel):
    insight_type: str = Field(description="Type of insight (trend, pattern, anomaly, etc.)")
    title: str = Field(description="Title of the insight")
    description: str = Field(description="Detailed description of the insight")
    data_points: List[Dict[str, Any]] = Field(description="Raw data points for visualization")
    significance: float = Field(description="Significance score (0-1)", ge=0, le=1)
    implications: List[str] = Field(description="Implications of this insight")

class VisualizationSpec(BaseModel):
    chart_type: str = Field(description="Type of chart (bar, line, pie, scatter, heatmap)")
    title: str = Field(description="Chart title")
    x_label: Optional[str] = Field(default=None, description="X-axis label")
    y_label: Optional[str] = Field(default=None, description="Y-axis label")
    data: Dict[str, Any] = Field(description="Data for the visualization")

class InsightAnalysis(BaseModel):
    research_query: str = Field(description="Original research query")
    analysis_timestamp: datetime = Field(default_factory=datetime.now)

    # Categorization insights
    source_categorization: List[CategoryDistribution] = Field(
        description="Distribution of sources by type"
    )
    topic_categorization: List[CategoryDistribution] = Field(
        description="Distribution of content by subtopic"
    )

    # Trend insights
    temporal_trends: List[TrendData] = Field(
        description="Trends over time if temporal data available"
    )
    relevance_trends: List[TrendData] = Field(
        description="Relevance score trends across sources"
    )

    # Statistical insights
    key_statistics: Dict[str, float] = Field(
        description="Key statistical metrics"
    )
    statistical_insights: List[StatisticalInsight] = Field(
        description="Detailed statistical insights"
    )

    # Visualizations
    visualizations: List[VisualizationSpec] = Field(
        description="Specifications for charts and graphs"
    )

    # Patterns and implications
    patterns_identified: List[str] = Field(
        description="Key patterns identified in the data"
    )
    future_implications: List[str] = Field(
        description="Future implications based on insights"
    )

    # Summary
    executive_insight_summary: str = Field(
        description="Executive summary of all insights"
    )

class ReportSection(BaseModel):
    title: str = Field(description="Section title")
    content: str = Field(description="Section content (markdown supported)")
    subsections: Optional[List['ReportSection']] = Field(
        default=None, description="Nested subsections"
    )
    visualizations: Optional[List[VisualizationSpec]] = Field(
        default=None, description="Associated visualizations"
    )
    citations: Optional[List[str]] = Field(
        default=None, description="Citations used in this section"
    )

class ResearchReport(BaseModel):
    title: str = Field(description="Report title")
    executive_summary: str = Field(description="Executive summary of entire research")
    report_timestamp: datetime = Field(default_factory=datetime.now)

    # Main report sections
    introduction: ReportSection = Field(description="Introduction section")
    methodology: ReportSection = Field(description="Research methodology section")
    findings: ReportSection = Field(description="Main findings section")
    analysis: ReportSection = Field(description="Critical analysis section")
    insights: ReportSection = Field(description="Statistical insights section")
    conclusions: ReportSection = Field(description="Conclusions and recommendations")

    # Supporting information
    appendices: Optional[List[ReportSection]] = Field(
        default=None, description="Additional appendices"
    )
    references: List[ResearchSource] = Field(
        description="All sources referenced in the report"
    )

    # Metadata
    word_count: int = Field(description="Total word count of report")
    key_takeaways: List[str] = Field(description="Bullet-point key takeaways")

    # Export formats
    export_formats_available: List[str] = Field(
        default=["pdf", "docx", "xml"],
        description="Available export formats"
    )

# Update forward reference
ReportSection.model_rebuild()

class ResearchState(BaseModel):
    research_query: str = Field(description="User's research question")
    search_results: Optional[WebSearchResult] = Field(
        default=None, description="Results from contextual retriever agent"
    )
    critical_analysis: Optional[CriticalAnalysis] = Field(
        default=None, description="Analysis from critical analysis agent"
    )
    insight_analysis: Optional[InsightAnalysis] = Field(
        default=None, description="Statistical insights from insight generation agent"
    )
    research_report: Optional[ResearchReport] = Field(
        default=None, description="Comprehensive research report from report builder agent"
    )
    current_step: str = Field(
        default="initial", description="Current step in the research process"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if any step fails"
    )