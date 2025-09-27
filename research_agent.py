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
    CriticalAnalysis, KeyFinding, Contradiction, SourceValidation,
    RefinedQuery, Subtopic
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
        self.query_parser = PydanticOutputParser(pydantic_object=RefinedQuery)

    def refine_query(self, query: str) -> RefinedQuery:
        """Interpret and refine broad or ambiguous queries into subtopics"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research query specialist. Your job is to:
1. Interpret broad or ambiguous research topics
2. Break them down into specific, searchable subtopics
3. Generate refined search queries for comprehensive coverage
4. Identify different source types needed (academic, news, reports, etc.)

{format_instructions}"""),
            ("human", """Research Query: {query}

Please analyze this query and:
1. Interpret what the user is really asking about
2. Break it down into 3-5 key subtopics
3. Generate specific search queries for each subtopic
4. Recommend search strategies for diverse sources""")
        ])

        try:
            formatted_prompt = prompt.format_messages(
                format_instructions=self.query_parser.get_format_instructions(),
                query=query
            )
            response = self.llm.invoke(formatted_prompt)
            return self.query_parser.parse(response.content)
        except Exception as e:
            print(f"Query refinement failed: {e}, using original query")
            # Fallback to simple refinement
            return RefinedQuery(
                original_query=query,
                interpretation=f"Researching information about {query}",
                refined_queries=[query, f"{query} latest developments", f"{query} analysis"],
                subtopics=[
                    Subtopic(
                        topic="General Overview",
                        description=f"General information about {query}",
                        search_queries=[query],
                        importance=0.8
                    )
                ],
                search_strategy="Standard web search across multiple source types"
            )

    def search_diverse_sources(self, query: str, source_type: str = None) -> List[Dict]:
        """Search across diverse source types with specialized queries"""
        results = []

        # Build search query with source type hints
        if source_type == "academic":
            search_query = f"{query} site:arxiv.org OR site:scholar.google.com OR site:pubmed.gov OR research paper"
        elif source_type == "news":
            search_query = f"{query} news latest site:reuters.com OR site:bloomberg.com OR site:wsj.com"
        elif source_type == "reports":
            search_query = f"{query} report analysis site:mckinsey.com OR site:deloitte.com OR industry report"
        elif source_type == "databases":
            search_query = f"{query} data statistics site:statista.com OR site:data.gov OR database"
        else:
            search_query = query

        try:
            tavily = TavilyClient(api_key=self.tavily_api_key)
            # Include more parameters for better results
            search_response = tavily.search(
                search_query,
                max_results=10,
                include_domains=self._get_domains_for_type(source_type),
                search_depth="advanced"  # More thorough search
            )

            for result in search_response.get("results", []):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("content", "")[:500],
                    "domain": result.get("url", "").split("/")[2] if "/" in result.get("url", "") else "",
                    "source_type": source_type or "general"
                })
        except Exception as e:
            print(f"Search for {source_type} sources failed: {e}")

        return results

    def _get_domains_for_type(self, source_type: str) -> List[str]:
        """Get relevant domains for each source type"""
        domain_map = {
            "academic": ["arxiv.org", "scholar.google.com", "pubmed.gov", "ieee.org", "nature.com"],
            "news": ["reuters.com", "bloomberg.com", "wsj.com", "ft.com", "economist.com"],
            "reports": ["mckinsey.com", "deloitte.com", "pwc.com", "gartner.com", "forrester.com"],
            "databases": ["statista.com", "data.gov", "worldbank.org", "oecd.org"]
        }
        return domain_map.get(source_type, [])

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
        print(f"Contextual Retriever Agent: Processing query: {query}")

        try:
            # Step 1: Refine and interpret the query
            print("Step 1: Refining query and generating subtopics...")
            refined_query = self.refine_query(query)

            # Step 2: Search diverse sources for each subtopic
            print(f"Step 2: Searching across {len(refined_query.subtopics)} subtopics...")
            all_results = []
            sources_by_subtopic = {}

            for subtopic in refined_query.subtopics:
                print(f"  Searching subtopic: {subtopic.topic}")
                subtopic_results = []

                # Search different source types for each subtopic
                for source_type in ["academic", "news", "reports", "general"]:
                    for search_query in subtopic.search_queries[:2]:  # Limit queries per subtopic
                        type_results = self.search_diverse_sources(search_query, source_type)
                        subtopic_results.extend(type_results)

                sources_by_subtopic[subtopic.topic] = subtopic_results
                all_results.extend(subtopic_results)

            # Step 3: Analyze and rank all results
            print(f"Step 3: Analyzing and ranking {len(all_results)} total results...")
            if not all_results:
                return {
                    "research_query": query,
                    "search_results": None,
                    "current_step": "error",
                    "error": "No search results found across any sources"
                }

            # Create enhanced search result with subtopic organization
            web_search_result = self.analyze_and_rank(query, all_results)
            web_search_result.refined_query = refined_query
            web_search_result.sources_by_subtopic = {
                topic: self._convert_to_research_sources(sources[:5])  # Top 5 per subtopic
                for topic, sources in sources_by_subtopic.items()
            }
            web_search_result.search_strategy = f"Multi-source search across {len(refined_query.subtopics)} subtopics with academic, news, and industry sources"

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
                "error": f"Enhanced search failed: {str(e)}"
            }

    def _convert_to_research_sources(self, sources: List[Dict]) -> List[ResearchSource]:
        """Convert raw search results to ResearchSource objects"""
        research_sources = []
        for i, source in enumerate(sources):
            source_type = SourceType.ACADEMIC_PAPER if "academic" in source.get("source_type", "") else \
                         SourceType.NEWS_ARTICLE if "news" in source.get("source_type", "") else \
                         SourceType.REPORT if "report" in source.get("source_type", "") else \
                         SourceType.WEBSITE

            research_sources.append(ResearchSource(
                title=source.get("title", ""),
                url=source.get("url", ""),
                snippet=source.get("snippet", ""),
                source_type=source_type,
                relevance_score=90 - (i * 10),  # Simple scoring
                domain=source.get("domain", ""),
                reasoning="Relevant to subtopic research",
                published_date=None,
                author=None
            ))
        return research_sources


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
1. Synthesize information from multiple sources across different subtopics
2. Identify contradictions and conflicts between sources
3. Validate source credibility based on source type (academic, news, reports)
4. Find consensus points and gaps
5. Provide actionable recommendations
6. Consider the diversity of sources (academic papers, news, industry reports, databases)

Analyze the search results thoroughly, considering the subtopic breakdown and source diversity.

{format_instructions}"""),
            ("human", """Research Query: {query}

Query Interpretation: {interpretation}
Number of Subtopics: {num_subtopics}
Subtopics: {subtopics}

Search Results to Analyze:
Total Sources: {total_sources}
Source Types Distribution: {source_types}

Sources by Subtopic:
{sources_by_subtopic}

All Sources:
{sources_detail}

Please provide a comprehensive critical analysis considering the subtopic organization and source diversity.""")
        ])

        # Prepare source details for analysis
        sources_detail = []
        source_type_counts = {}

        for i, source in enumerate(search_results.sources, 1):
            source_type_counts[source.source_type.value] = source_type_counts.get(source.source_type.value, 0) + 1
            sources_detail.append(f"""
Source {i}: {source.title}
- Type: {source.source_type.value}
- Relevance: {source.relevance_score}%
- URL: {source.url}
- Content: {source.snippet}
- Reasoning: {source.reasoning}
""")

        # Prepare subtopic breakdown
        subtopics_str = ""
        sources_by_subtopic_str = ""

        if search_results.refined_query:
            subtopics_str = ", ".join([st.topic for st in search_results.refined_query.subtopics])

            if search_results.sources_by_subtopic:
                for topic, sources in search_results.sources_by_subtopic.items():
                    sources_by_subtopic_str += f"\n{topic}:\n"
                    for s in sources[:3]:  # Top 3 per subtopic
                        sources_by_subtopic_str += f"  - {s.title} ({s.source_type.value})\n"

        try:
            formatted_prompt = prompt.format_messages(
                format_instructions=self.parser.get_format_instructions(),
                query=query,
                interpretation=search_results.refined_query.interpretation if search_results.refined_query else query,
                num_subtopics=len(search_results.refined_query.subtopics) if search_results.refined_query else 1,
                subtopics=subtopics_str or "Single topic",
                total_sources=len(search_results.sources),
                source_types=", ".join([f"{k}: {v}" for k, v in source_type_counts.items()]),
                sources_by_subtopic=sources_by_subtopic_str or "N/A",
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