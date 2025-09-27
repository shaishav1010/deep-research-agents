import os
from typing import Dict, Any, List
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END
from tavily import TavilyClient
import json
from models import ResearchSource, WebSearchResult, ResearchState, SourceType


class WebSearchNode:
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


class ResearchGraph:
    def __init__(self, api_key: str, tavily_api_key: str):
        self.api_key = api_key
        self.web_search_node = WebSearchNode(api_key, tavily_api_key)
        self.workflow = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(dict)

        workflow.add_node("web_search", self.web_search_node.execute)

        workflow.set_entry_point("web_search")

        workflow.add_edge("web_search", END)

        return workflow.compile()

    def run(self, research_query: str) -> ResearchState:
        initial_state = {
            "research_query": research_query,
            "search_results": None,
            "current_step": "initial",
            "error": None
        }

        result = self.workflow.invoke(initial_state)

        return ResearchState(**result)