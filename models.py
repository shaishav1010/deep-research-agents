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


class ResearchState(BaseModel):
    research_query: str = Field(description="User's research question")
    search_results: Optional[WebSearchResult] = Field(
        default=None, description="Results from web search"
    )
    current_step: str = Field(
        default="initial", description="Current step in the research process"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if any step fails"
    )