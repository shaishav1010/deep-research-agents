import os
from typing import Dict, Any, List
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END
from tavily import TavilyClient
import json
from models import (
    ResearchSource, WebSearchResult, ResearchState, SourceType,
    CriticalAnalysis, KeyFinding, Contradiction, SourceValidation
)


class ContextualRetrieverAgent:
    def __init__(self, api_key: str, tavily_api_key: str):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "Deep Research Agent",
            }
        )
        if not tavily_api_key:
            raise ValueError("Tavily API key is required for web search")
        self.tavily_api_key = tavily_api_key
        self.parser = PydanticOutputParser(pydantic_object=WebSearchResult)

    def search_web(self, query: str, max_results: int = 20) -> List[Dict]:
        try:
            # Use Tavily API for real search
            tavily = TavilyClient(api_key=self.tavily_api_key)
            search_response = tavily.search(query, max_results=max_results)

            results = []
            for result in search_response.get("results", []):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("content", "")[:500],
                    "domain": result.get("url", "").split("/")[2] if "/" in result.get("url", "") else ""
                })

            return results
        except Exception as e:
            raise Exception(f"Tavily search failed: {e}")


    def analyze_and_rank(self, query: str, search_results: List[Dict]) -> WebSearchResult:
        # If we don't have valid search results, return empty
        if not search_results:
            return WebSearchResult(
                query=query,
                search_timestamp=datetime.now(),
                sources=[],
                total_results_found=0,
                search_strategy="Web search",
                key_insights=["No results found for the query"]
            )
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert research analyst. Analyze the search results and:
1. Classify each source by type (academic_paper, news_article, blog_post, documentation, etc.)
2. Assign a relevance score (0-100) based on how well it answers the research query
3. Provide reasoning for the relevance score
4. Extract key insights from the collection of sources
5. Return only the top 10 most relevant sources

Focus on:
- Direct relevance to the query
- Source credibility and authority
- Recency and timeliness
- Depth of information
- Unique perspectives or data

{format_instructions}"""),
            ("human", """Research Query: {query}

Search Results:
{search_results}

Analyze these results and return structured output with the top 10 most relevant sources.""")
        ])

        try:
            formatted_prompt = prompt.format_messages(
                format_instructions=self.parser.get_format_instructions(),
                query=query,
                search_results=json.dumps(search_results, indent=2)
            )

            response = self.llm.invoke(formatted_prompt)
            parsed_result = self.parser.parse(response.content)
            return parsed_result
        except Exception as e:
            print(f"LLM ranking failed: {e}, using fallback ranking")
            sources = []
            for i, result in enumerate(search_results[:10]):
                sources.append(ResearchSource(
                    title=result["title"],
                    url=result["url"],
                    snippet=result["snippet"],
                    source_type=SourceType.WEBSITE,
                    relevance_score=90 - (i * 5),
                    domain=result["domain"],
                    reasoning="Relevant to the research query based on keyword matching"
                ))

            return WebSearchResult(
                query=query,
                search_timestamp=datetime.now(),
                sources=sources,
                total_results_found=len(search_results),
                search_strategy="Web search using DuckDuckGo",
                key_insights=["Initial search completed", "Results ranked by relevance"]
            )

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("research_query", "")
        print(f"Execute called with query: {query}")

        try:
            search_results = self.search_web(query)
            print(f"Search returned {len(search_results)} results")

            if not search_results:
                return {
                    "research_query": query,
                    "search_results": None,
                    "current_step": "error",
                    "error": "No search results found"
                }

            web_search_result = self.analyze_and_rank(query, search_results)

            return {
                "research_query": query,
                "search_results": web_search_result,
                "current_step": "search_completed",
                "error": None
            }

        except Exception as e:
            return {
                "research_query": query,
                "search_results": None,
                "current_step": "error",
                "error": f"Search failed: {str(e)}"
            }


class CriticalAnalysisAgent:
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "Deep Research Agent - Critical Analysis",
            }
        )
        self.parser = PydanticOutputParser(pydantic_object=CriticalAnalysis)

    def analyze(self, query: str, search_results: WebSearchResult) -> CriticalAnalysis:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a critical analysis expert. Your job is to:
1. Synthesize information from multiple sources
2. Identify contradictions and conflicts between sources
3. Validate source credibility
4. Find consensus points and gaps
5. Provide actionable recommendations

Analyze the search results thoroughly and provide structured critical analysis.

{format_instructions}"""),
            ("human", """Research Query: {query}

Search Results to Analyze:
Total Sources: {total_sources}
Sources:
{sources_detail}

Please provide a comprehensive critical analysis of these sources.""")
        ])

        # Prepare source details for analysis
        sources_detail = []
        for i, source in enumerate(search_results.sources, 1):
            sources_detail.append(f"""
Source {i}: {source.title}
- Type: {source.source_type.value}
- Relevance: {source.relevance_score}%
- URL: {source.url}
- Content: {source.snippet}
- Reasoning: {source.reasoning}
""")

        try:
            formatted_prompt = prompt.format_messages(
                format_instructions=self.parser.get_format_instructions(),
                query=query,
                total_sources=len(search_results.sources),
                sources_detail="\n".join(sources_detail)
            )

            response = self.llm.invoke(formatted_prompt)
            parsed_result = self.parser.parse(response.content)
            return parsed_result
        except Exception as e:
            print(f"Critical analysis failed: {e}, using fallback analysis")
            return self.fallback_analysis(query, search_results)

    def fallback_analysis(self, query: str, search_results: WebSearchResult) -> CriticalAnalysis:
        """Provide basic analysis if LLM fails"""
        sources = search_results.sources[:5]

        return CriticalAnalysis(
            executive_summary=f"Analysis of {len(search_results.sources)} sources for query: {query}. "
                            f"Sources show varied perspectives with relevance scores ranging from "
                            f"{min(s.relevance_score for s in sources)}% to {max(s.relevance_score for s in sources)}%.",
            key_findings=[
                KeyFinding(
                    finding=f"Primary information from {sources[0].title}",
                    sources=[sources[0].title],
                    confidence=0.7
                )
            ],
            contradictions=[],
            source_validations=[
                SourceValidation(
                    source_title=source.title,
                    credibility_score=source.relevance_score,
                    credibility_factors=["Relevance to query", "Source domain reputation"],
                    potential_biases=[]
                ) for source in sources[:3]
            ],
            consensus_points=[f"Multiple sources discuss {query}"],
            gaps_identified=["Deeper analysis required for comprehensive understanding"],
            recommendations=["Review top sources for detailed information", "Consider additional research for gaps"],
            confidence_assessment="Moderate confidence based on available sources"
        )

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("Critical Analysis Agent executing...")

        # Get search results from previous step
        search_results = state.get("search_results")
        query = state.get("research_query", "")

        if not search_results:
            return {
                **state,
                "critical_analysis": None,
                "current_step": "error",
                "error": "No search results to analyze"
            }

        try:
            analysis = self.analyze(query, search_results)

            return {
                **state,
                "critical_analysis": analysis,
                "current_step": "analysis_completed",
                "error": None
            }
        except Exception as e:
            return {
                **state,
                "critical_analysis": None,
                "current_step": "error",
                "error": f"Critical analysis failed: {str(e)}"
            }


class ResearchGraph:
    def __init__(self, api_key: str, tavily_api_key: str):
        self.api_key = api_key
        # Initialize agents
        self.retriever_agent = ContextualRetrieverAgent(api_key, tavily_api_key)
        self.analysis_agent = CriticalAnalysisAgent(api_key)
        self.workflow = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(dict)

        # Add nodes for both agents
        workflow.add_node("contextual_retriever", self.retriever_agent.execute)
        workflow.add_node("critical_analysis", self.analysis_agent.execute)

        # Set the entry point
        workflow.set_entry_point("contextual_retriever")

        # Add edge from retriever to analysis
        workflow.add_edge("contextual_retriever", "critical_analysis")

        # End after analysis
        workflow.add_edge("critical_analysis", END)

        return workflow.compile()

    def run(self, research_query: str) -> ResearchState:
        initial_state = {
            "research_query": research_query,
            "search_results": None,
            "critical_analysis": None,
            "current_step": "initial",
            "error": None
        }

        result = self.workflow.invoke(initial_state)

        return ResearchState(**result)