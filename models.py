from pydantic import BaseModel, Field
from typing import List, Optional
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


class WebSearchResult(BaseModel):
    query: str = Field(description="Original research query")
    search_timestamp: datetime = Field(
        default_factory=datetime.now, description="When the search was performed"
    )
    sources: List[ResearchSource] = Field(
        description="List of top sources ranked by relevance"
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

class ResearchState(BaseModel):
    research_query: str = Field(description="User's research question")
    search_results: Optional[WebSearchResult] = Field(
        default=None, description="Results from contextual retriever agent"
    )
    critical_analysis: Optional[CriticalAnalysis] = Field(
        default=None, description="Analysis from critical analysis agent"
    )
    current_step: str = Field(
        default="initial", description="Current step in the research process"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if any step fails"
    )