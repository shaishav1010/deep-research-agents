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
    RefinedQuery, Subtopic, InsightAnalysis, CategoryDistribution,
    TrendData, StatisticalInsight, VisualizationSpec,
    ResearchReport, ReportSection
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
                max_results=3,  # Reduced for faster testing
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

            # Safely process search results with proper error handling
            if search_results and len(search_results) > 0:
                for i, result in enumerate(search_results[:10]):
                    try:
                        sources.append(ResearchSource(
                            title=result.get("title", "Unknown Title"),
                            url=result.get("url", ""),
                            snippet=result.get("snippet", "No snippet available"),
                            source_type=SourceType.WEBSITE,
                            relevance_score=90 - (i * 5),
                            domain=result.get("domain", "unknown"),
                            reasoning="Relevant to the research query based on keyword matching"
                        ))
                    except Exception as source_error:
                        print(f"Error processing source {i}: {source_error}")
                        continue

            # Always return a valid result, even with empty sources
            return WebSearchResult(
                query=query,
                search_timestamp=datetime.now(),
                sources=sources,
                total_results_found=len(search_results) if search_results else 0,
                search_strategy="Web search with fallback ranking",
                key_insights=[f"Found {len(sources)} sources" if sources else "No sources found"]
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
            # Continue even with no results - let downstream agents handle it
            if not all_results:
                print("Warning: No search results found, creating empty result set")
                # Create an empty but valid result
                empty_result = WebSearchResult(
                    query=query,
                    search_timestamp=datetime.now(),
                    sources=[],
                    total_results_found=0,
                    search_strategy="No results available",
                    key_insights=["No search results found - using sample data for demonstration"]
                )
                empty_result.refined_query = refined_query
                empty_result.sources_by_subtopic = {}

                return {
                    "research_query": query,
                    "search_results": empty_result,
                    "current_step": "search_completed",
                    "error": None
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


class InsightGenerationAgent:
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "Deep Research Agent - Insight Generation",
            }
        )
        self.parser = PydanticOutputParser(pydantic_object=InsightAnalysis)

    def generate_insights(self, query: str, search_results: WebSearchResult,
                         critical_analysis: CriticalAnalysis = None) -> InsightAnalysis:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data insights and visualization expert specializing in extracting quantitative insights from research data.

Your primary task is to:
1. Extract SPECIFIC QUANTITATIVE DATA from the research content that directly relates to the query
2. Create visualizations that show ACTUAL DATA POINTS found in the sources, not just metadata about sources
3. Generate EXACTLY 3 meaningful charts/visualizations that best answer the user's query

For example:
- If the query is about "mobile phone impact on kids by age", extract actual percentages, usage rates, impact metrics BY AGE GROUP
- If the query is about "EV market trends", extract actual market share numbers, growth percentages, regional data
- If the query is about "climate change effects", extract temperature changes, sea level rises, timeline data

IMPORTANT RULES:
- Focus on CONTENT DATA not source metadata
- Extract real numbers, percentages, categories from the research findings
- Create charts that directly answer the user's question
- Maximum 3 visualizations, each must be highly relevant
- If specific data isn't found, indicate what data would be needed

{format_instructions}"""),
            ("human", """Research Query: {query}

Critical Analysis Key Findings:
{key_findings_detail}

Consensus Points from Research:
{consensus_detail}

All Research Content and Data:
{sources_content}

INSTRUCTIONS:
1. Identify the main data categories relevant to "{query}"
2. Extract specific numbers, percentages, or categorical data from the sources
3. Create 3 visualizations that best represent this data:
   - Chart 1: Primary data relationship (e.g., age groups vs usage)
   - Chart 2: Secondary insights (e.g., impact metrics, trends)
   - Chart 3: Comparative or distribution data
4. For each visualization, use ACTUAL DATA from the sources, not generic placeholders
5. If the query asks for specific breakdowns (age groups, regions, time periods), ensure charts reflect these

Remember: Users want to see the ACTUAL RESEARCH DATA visualized, not information about the sources themselves.""")
        ])

        # Prepare detailed content from sources
        sources_content = []
        for i, source in enumerate(search_results.sources, 1):
            sources_content.append(f"""
Source {i}: {source.title}
Content: {source.snippet}
Type: {source.source_type.value}
""")

        # Prepare key findings detail (if critical analysis available)
        key_findings_detail = ""
        consensus_detail = ""
        if critical_analysis:
            key_findings_detail = "\n".join([
                f"- {finding.finding} (Confidence: {finding.confidence:.0%})"
                for finding in critical_analysis.key_findings
            ])
            consensus_detail = "\n".join([
                f"- {point}"
                for point in critical_analysis.consensus_points
            ])

        try:
            formatted_prompt = prompt.format_messages(
                format_instructions=self.parser.get_format_instructions(),
                query=query,
                key_findings_detail=key_findings_detail,
                consensus_detail=consensus_detail,
                sources_content="\n".join(sources_content)
            )

            response = self.llm.invoke(formatted_prompt)
            return self.parser.parse(response.content)

        except Exception as e:
            # Return fallback insights
            return self._generate_fallback_insights(query, search_results, critical_analysis)

    def _generate_fallback_insights(self, query: str, search_results: WebSearchResult,
                                   critical_analysis: CriticalAnalysis = None) -> InsightAnalysis:
        """Generate contextual insights based on query without LLM"""
        import re

        # Analyze query to determine what kind of data to extract
        query_lower = query.lower()

        # Build source type counts
        source_type_counts = {}
        for source in search_results.sources:
            source_type = source.source_type.value
            source_type_counts[source_type] = source_type_counts.get(source_type, 0) + 1

        # Calculate source categorization
        total_sources = len(search_results.sources)
        source_categories = []
        for source_type, count in source_type_counts.items():
            source_categories.append(CategoryDistribution(
                category=source_type,
                count=count,
                percentage=(count / total_sources) * 100 if total_sources > 0 else 0,
                description=f"Sources categorized as {source_type}"
            ))

        # Calculate topic categorization if subtopics exist
        topic_categories = []
        if search_results.sources_by_subtopic:
            total_by_topic = sum(len(sources) for sources in search_results.sources_by_subtopic.values())
            for topic, sources in search_results.sources_by_subtopic.items():
                topic_categories.append(CategoryDistribution(
                    category=topic,
                    count=len(sources),
                    percentage=(len(sources) / total_by_topic) * 100 if total_by_topic > 0 else 0,
                    description=f"Sources related to {topic}"
                ))

        # Generate relevance trends
        relevance_trends = []
        for i, source in enumerate(search_results.sources[:10], 1):
            relevance_trends.append(TrendData(
                label=f"Source {i}",
                value=source.relevance_score,
                category="Relevance Score"
            ))

        # Key statistics
        relevance_scores = [s.relevance_score for s in search_results.sources]
        key_statistics = {
            "avg_relevance": sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0,
            "max_relevance": max(relevance_scores) if relevance_scores else 0,
            "min_relevance": min(relevance_scores) if relevance_scores else 0,
            "total_sources": total_sources,
            "num_subtopics": len(search_results.refined_query.subtopics) if search_results.refined_query else 1,
            "confidence_level": 0.65 if search_results.sources else 0.5  # Based on search results only
        }

        # Create contextual visualizations based on query type and extracted data
        visualizations = []

        # Extract numeric data from source snippets
        def extract_numbers_from_sources(sources, pattern=r'\b(\d+(?:\.\d+)?)\s*(%|percent|percentage|years?|months?|days?|hours?)?'):
            """Extract numbers with optional units from source snippets"""
            extracted_data = []
            for source in sources:
                matches = re.findall(pattern, source.snippet, re.IGNORECASE)
                for match in matches:
                    number = float(match[0])
                    unit = match[1] if match[1] else ''
                    extracted_data.append({
                        'value': number,
                        'unit': unit.lower(),
                        'source': source.title,
                        'snippet': source.snippet[:100]
                    })
            return extracted_data

        # Extract age-related data
        def extract_age_data(sources):
            """Extract age-related statistics from sources"""
            age_patterns = [
                (r'(\d+)\s*(?:to|-)\s*(\d+)\s*(?:year|yr)', 'age_range'),
                (r'age(?:d|s)?\s*(\d+)', 'specific_age'),
                (r'(toddler|infant|child|teen|adolescent|adult)', 'age_group'),
                (r'(\d+)\s*(?:%|percent)\s*(?:of\s+)?(?:children|teens|adults)', 'percentage')
            ]

            age_data = {}
            for source in sources:
                text = source.snippet.lower()
                for pattern, data_type in age_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        if data_type not in age_data:
                            age_data[data_type] = []
                        age_data[data_type].extend(matches)

            return age_data

        print(f"Creating visualizations for query with {len(search_results.sources)} sources")

        # Extract all numeric data from sources
        numeric_data = extract_numbers_from_sources(search_results.sources)

        # Determine visualization type based on query keywords and extracted data
        if any(keyword in query_lower for keyword in ['age', 'group', 'demographic', 'kids', 'children', 'teens', 'brain']):
            age_data = extract_age_data(search_results.sources)

            # Try to extract percentage data by age groups
            percentages_by_age = []
            age_groups = []

            for source in search_results.sources:
                # Look for patterns like "X% of children/teens"
                matches = re.findall(r'(\d+(?:\.\d+)?)\s*%\s*of\s*(children|teens|adults|toddlers|infants)',
                                   source.snippet, re.IGNORECASE)
                for match in matches:
                    percentages_by_age.append(float(match[0]))
                    age_groups.append(match[1].capitalize())

            # If we found age-related data, use it
            if percentages_by_age and age_groups:
                visualizations.append(VisualizationSpec(
                    chart_type="bar",
                    title="Statistics by Age Group (From Sources)",
                    x_label="Age Groups",
                    y_label="Percentage (%)",
                    data={
                        "x": age_groups[:5],  # Limit to 5 groups
                        "y": percentages_by_age[:5]
                    }
                ))
            else:
                # Use any percentages found in the context of the query
                percentage_values = [d['value'] for d in numeric_data if d['unit'] in ['%', 'percent', 'percentage']]
                if percentage_values:
                    visualizations.append(VisualizationSpec(
                        chart_type="bar",
                        title="Key Statistics from Research",
                        x_label="Data Points",
                        y_label="Value (%)",
                        data={
                            "x": [f"Statistic {i+1}" for i in range(min(5, len(percentage_values)))],
                            "y": percentage_values[:5]
                        }
                    ))

        elif any(keyword in query_lower for keyword in ['trend', 'over time', 'year', 'timeline', 'history']):
            # Extract year data
            year_pattern = r'\b(19\d{2}|20\d{2})\b'
            years_data = {}

            for source in search_results.sources:
                years = re.findall(year_pattern, source.snippet)
                for year in years:
                    # Look for associated values near the year
                    value_pattern = rf'{year}\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(%|percent)?'
                    values = re.findall(value_pattern, source.snippet, re.IGNORECASE)
                    if values:
                        years_data[year] = float(values[0][0])

            if years_data:
                sorted_years = sorted(years_data.items())
                visualizations.append(VisualizationSpec(
                    chart_type="line",
                    title="Trends Over Time (Extracted from Sources)",
                    x_label="Year",
                    y_label="Value",
                    data={
                        "x": [year for year, _ in sorted_years],
                        "y": [value for _, value in sorted_years]
                    }
                ))
            elif numeric_data:
                # Use any numeric progression found
                values = [d['value'] for d in numeric_data[:6]]
                if values:
                    visualizations.append(VisualizationSpec(
                        chart_type="line",
                        title="Data Progression",
                        x_label="Data Points",
                        y_label="Value",
                        data={
                            "x": [f"Point {i+1}" for i in range(len(values))],
                            "y": values
                        }
                    ))

        elif any(keyword in query_lower for keyword in ['compare', 'versus', 'vs', 'difference', 'comparison']):
            # Extract comparison data
            comparison_values = []
            comparison_labels = []

            for source in search_results.sources:
                # Look for patterns like "A is X% while B is Y%"
                comp_pattern = r'(\w+)\s+(?:is|has|shows)\s+(\d+(?:\.\d+)?)\s*%'
                matches = re.findall(comp_pattern, source.snippet, re.IGNORECASE)
                for match in matches:
                    comparison_labels.append(match[0][:20])  # Limit label length
                    comparison_values.append(float(match[1]))

            if comparison_values and comparison_labels:
                visualizations.append(VisualizationSpec(
                    chart_type="bar",
                    title="Comparative Analysis (From Sources)",
                    x_label="Categories",
                    y_label="Values (%)",
                    data={
                        "x": comparison_labels[:4],
                        "y": comparison_values[:4]
                    }
                ))
            elif numeric_data:
                # Use top numeric values for comparison
                top_values = sorted(numeric_data, key=lambda x: x['value'], reverse=True)[:4]
                if top_values:
                    visualizations.append(VisualizationSpec(
                        chart_type="bar",
                        title="Key Metrics Comparison",
                        x_label="Metrics",
                        y_label="Values",
                        data={
                            "x": [f"Metric {i+1}" for i in range(len(top_values))],
                            "y": [d['value'] for d in top_values]
                        }
                    ))

        # Add subtopic distribution if we have multiple subtopics
        if topic_categories and len(topic_categories) > 1:
            visualizations.append(VisualizationSpec(
                chart_type="pie",
                title="Research Coverage by Topic",
                data={
                    "labels": [cat.category for cat in topic_categories[:5]],  # Top 5 topics
                    "values": [cat.count for cat in topic_categories[:5]]
                }
            ))

        # Add relevance score visualization for top sources
        if search_results.sources:
            top_sources = search_results.sources[:5]
            source_relevances = [s.relevance_score for s in top_sources]
            source_labels = [f"Source {i+1}" for i in range(len(top_sources))]

            visualizations.append(VisualizationSpec(
                chart_type="bar",
                title="Top Source Relevance Scores",
                x_label="Research Sources",
                y_label="Relevance Score (%)",
                data={
                    "x": source_labels,
                    "y": source_relevances  # Already in percentage format
                }
            ))

        # If no visualizations created yet, but we have numeric data, use it
        if len(visualizations) == 0 and numeric_data:
            print("Creating visualization from extracted numeric data")
            top_numbers = sorted(numeric_data, key=lambda x: x['value'], reverse=True)[:5]
            if top_numbers:
                visualizations.append(VisualizationSpec(
                    chart_type="bar",
                    title="Key Quantitative Findings",
                    x_label="Data Points",
                    y_label="Values",
                    data={
                        "x": [f"Finding {i+1}" for i in range(len(top_numbers))],
                        "y": [d['value'] for d in top_numbers]
                    }
                ))

        # Final fallback if still no visualizations
        if len(visualizations) == 0:
            print("No extractable numeric data found in sources - showing source quality metrics")
            # Show source quality distribution
            quality_ranges = {"High (80-100)": 0, "Medium (60-79)": 0, "Low (40-59)": 0, "Very Low (<40)": 0}
            for score in relevance_scores:
                if score >= 80:
                    quality_ranges["High (80-100)"] += 1
                elif score >= 60:
                    quality_ranges["Medium (60-79)"] += 1
                elif score >= 40:
                    quality_ranges["Low (40-59)"] += 1
                else:
                    quality_ranges["Very Low (<40)"] += 1

            visualizations.append(VisualizationSpec(
                chart_type="bar",
                title="Source Quality Distribution",
                x_label="Quality Range",
                y_label="Number of Sources",
                data={
                    "x": list(quality_ranges.keys()),
                    "y": list(quality_ranges.values())
                }
            ))

        # Ensure we always have exactly 3 visualizations
        if len(visualizations) < 3:
            # Add source type distribution
            if len(visualizations) < 3 and source_categories:
                visualizations.append(VisualizationSpec(
                    chart_type="pie",
                    title="Source Type Distribution",
                    data={
                        "labels": [cat.category for cat in source_categories],
                        "values": [cat.count for cat in source_categories]
                    }
                ))

            # Add relevance score distribution
            if len(visualizations) < 3:
                relevance_bins = {"0-25": 0, "25-50": 0, "50-75": 0, "75-100": 0}
                for score in relevance_scores:
                    if score < 25:
                        relevance_bins["0-25"] += 1
                    elif score < 50:
                        relevance_bins["25-50"] += 1
                    elif score < 75:
                        relevance_bins["50-75"] += 1
                    else:
                        relevance_bins["75-100"] += 1

                visualizations.append(VisualizationSpec(
                    chart_type="bar",
                    title="Relevance Score Distribution",
                    x_label="Score Range",
                    y_label="Number of Sources",
                    data={
                        "x": list(relevance_bins.keys()),
                        "y": list(relevance_bins.values())
                    }
                ))

        # Ensure exactly 3 visualizations
        visualizations = visualizations[:3]

        # Debug: Verify visualization data
        for i, viz in enumerate(visualizations, 1):
            if viz.chart_type == "pie":
                print(f"Visualization {i}: {viz.title} (pie) - labels: {len(viz.data.get('labels', []))}, values: {len(viz.data.get('values', []))}")
            else:
                print(f"Visualization {i}: {viz.title} ({viz.chart_type}) - x: {len(viz.data.get('x', []))}, y: {len(viz.data.get('y', []))}")
                if len(viz.data.get('y', [])) > 0:
                    print(f"  Sample data: {viz.data.get('y', [])[:3]}...")

        # Statistical insights
        statistical_insights = [
            StatisticalInsight(
                insight_type="pattern",
                title="Source Diversity",
                description=f"Research includes {len(source_type_counts)} different source types, "
                           f"indicating {'diverse' if len(source_type_counts) >= 3 else 'limited'} perspective coverage",
                data_points=[{"type": k, "count": v} for k, v in source_type_counts.items()],
                significance=0.8 if len(source_type_counts) >= 3 else 0.5,
                implications=[
                    "Multiple source types provide balanced perspective" if len(source_type_counts) >= 3
                    else "Consider expanding source diversity"
                ]
            )
        ]

        if len(relevance_scores) > 0:
            statistical_insights.append(StatisticalInsight(
                insight_type="trend",
                title="Relevance Distribution",
                description=f"Average relevance score of {key_statistics['avg_relevance']:.1f}% "
                           f"with range from {key_statistics['min_relevance']:.0f}% to {key_statistics['max_relevance']:.0f}%",
                data_points=[{"source": i, "score": score} for i, score in enumerate(relevance_scores)],
                significance=0.7,
                implications=[
                    f"{'High' if key_statistics['avg_relevance'] > 70 else 'Moderate'} overall source quality",
                    "Top sources show strong alignment with research query"
                ]
            ))

        return InsightAnalysis(
            research_query=query,
            source_categorization=source_categories,
            topic_categorization=topic_categories,
            temporal_trends=[],  # No temporal data in fallback
            relevance_trends=relevance_trends,
            key_statistics=key_statistics,
            statistical_insights=statistical_insights,
            visualizations=visualizations,
            patterns_identified=[
                f"Identified {len(search_results.refined_query.subtopics) if search_results.refined_query else 1} main subtopics",
                f"Found {total_sources} relevant sources across multiple domains",
                f"Source quality shows {'consistent' if key_statistics['avg_relevance'] > 70 else 'varied'} relevance"
            ],
            future_implications=[
                "Additional research may uncover more specialized sources",
                "Cross-referencing findings could strengthen conclusions"
            ],
            executive_insight_summary=f"Statistical analysis of {total_sources} sources reveals "
                                    f"{'diverse' if len(source_type_counts) >= 3 else 'focused'} coverage "
                                    f"with average relevance of {key_statistics['avg_relevance']:.0f}%. "
                                    f"Key patterns show {len(source_type_counts)} source types and "
                                    f"{len(topic_categories)} subtopic areas."
        )

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("Insight Generation Agent executing...")

        search_results = state.get("search_results")
        critical_analysis = state.get("critical_analysis")  # Optional, not required
        query = state.get("research_query", "")

        if not search_results:
            return {
                **state,
                "insight_analysis": None,
                "current_step": "error",
                "error": "Missing search results for insight generation"
            }

        try:
            # Pass critical_analysis as optional parameter (can be None)
            insights = self.generate_insights(query, search_results, critical_analysis)

            # Debug logging
            print(f"Generated {len(insights.visualizations)} visualizations")
            for viz in insights.visualizations:
                print(f"  - {viz.chart_type}: {viz.title}")
                if viz.chart_type == "pie":
                    print(f"    Labels: {viz.data.get('labels', [])}")
                    print(f"    Values: {viz.data.get('values', [])}")
                else:
                    print(f"    X data: {viz.data.get('x', [])}")
                    print(f"    Y data: {viz.data.get('y', [])}")

            return {
                **state,
                "insight_analysis": insights,
                "current_step": "insights_completed",
                "error": None
            }
        except Exception as e:
            print(f"Insight generation error: {str(e)}")
            return {
                **state,
                "insight_analysis": None,
                "current_step": "error",
                "error": f"Insight generation failed: {str(e)}"
            }


class ReportBuilderAgent:
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "Deep Research Agent",
            }
        )
        self.parser = PydanticOutputParser(pydantic_object=ResearchReport)

    def generate_report(self, query: str, search_results: WebSearchResult,
                        critical_analysis: CriticalAnalysis,
                        insight_analysis: InsightAnalysis) -> ResearchReport:
        """Generate comprehensive research report from all previous analyses"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional research report writer. Your task is to compile all research findings into a comprehensive, well-structured report.

The report should:
1. Have a clear executive summary
2. Explain the research methodology
3. Present findings in a logical order
4. Include critical analysis
5. Present statistical insights
6. Provide actionable conclusions

{format_instructions}"""),
            ("human", """Generate a comprehensive research report for the following:

Research Query: {query}

Search Results Summary:
- Total sources found: {total_sources}
- Key source types: {source_types}
- Main domains: {domains}

Critical Analysis:
{critical_analysis}

Statistical Insights:
{insights}

Create a professional report with proper sections, clear takeaways, and word count.""")
        ])

        # Prepare source information
        source_types = list(set([s.source_type for s in search_results.sources])) if search_results.sources else []
        domains = list(set([s.domain for s in search_results.sources[:5]])) if search_results.sources else []

        chain = prompt | self.llm | self.parser

        try:
            report = chain.invoke({
                "query": query,
                "format_instructions": self.parser.get_format_instructions(),
                "total_sources": len(search_results.sources) if search_results.sources else 0,
                "source_types": ", ".join(source_types[:3]),
                "domains": ", ".join(domains[:3]),
                "critical_analysis": json.dumps({
                    "executive_summary": critical_analysis.executive_summary,
                    "key_findings": [f.finding for f in critical_analysis.key_findings[:3]],
                    "consensus_points": critical_analysis.consensus_points[:3],
                    "recommendations": critical_analysis.recommendations[:3]
                }, indent=2),
                "insights": json.dumps({
                    "key_statistics": insight_analysis.key_statistics,
                    "patterns": insight_analysis.patterns_identified[:3],
                    "executive_summary": insight_analysis.executive_insight_summary
                }, indent=2) if insight_analysis else "No insights available"
            })

            return report
        except Exception as e:
            print(f"LLM parsing failed: {e}")
            return self._generate_fallback_report(query, search_results, critical_analysis, insight_analysis)

    def _generate_fallback_report(self, query: str, search_results: WebSearchResult,
                                  critical_analysis: CriticalAnalysis,
                                  insight_analysis: InsightAnalysis) -> ResearchReport:
        """Generate fallback report if LLM fails"""

        # Count words in all content
        word_count = len(critical_analysis.executive_summary.split())
        word_count += sum(len(f.finding.split()) for f in critical_analysis.key_findings)

        # Extract key takeaways
        key_takeaways = []
        if critical_analysis.key_findings:
            key_takeaways.extend([f.finding for f in critical_analysis.key_findings[:3]])
        if critical_analysis.recommendations:
            key_takeaways.extend(critical_analysis.recommendations[:2])

        # Create report sections
        introduction = ReportSection(
            title="Introduction",
            content=f"This report presents a comprehensive analysis of: {query}\n\n"
                   f"The research was conducted using multiple sources and analytical techniques to provide "
                   f"a thorough understanding of the topic.",
            citations=[s.title for s in search_results.sources[:2]] if search_results.sources else []
        )

        methodology = ReportSection(
            title="Research Methodology",
            content=f"The research methodology consisted of:\n\n"
                   f"1. **Query Refinement**: Breaking down the query into subtopics\n"
                   f"2. **Multi-source Search**: Searching across {len(search_results.sources)} sources\n"
                   f"3. **Critical Analysis**: Evaluating source credibility and identifying patterns\n"
                   f"4. **Statistical Analysis**: Generating insights and visualizations\n"
                   f"5. **Report Compilation**: Synthesizing all findings into this comprehensive report"
        )

        findings = ReportSection(
            title="Key Findings",
            content="\n\n".join([
                f"**Finding {i+1}**: {f.finding}\n*Confidence: {f.confidence:.0%}*"
                for i, f in enumerate(critical_analysis.key_findings[:5])
            ]) if critical_analysis.key_findings else "No specific findings identified.",
            visualizations=insight_analysis.visualizations[:2] if insight_analysis else None
        )

        analysis = ReportSection(
            title="Critical Analysis",
            content=f"## Executive Summary\n{critical_analysis.executive_summary}\n\n"
                   f"## Consensus Points\n" + "\n".join([
                       f"- {point}" for point in critical_analysis.consensus_points[:5]
                   ]) + "\n\n## Knowledge Gaps\n" + "\n".join([
                       f"- {gap}" for gap in critical_analysis.gaps_identified[:3]
                   ])
        )

        insights = ReportSection(
            title="Statistical Insights",
            content=f"## Key Metrics\n" + "\n".join([
                f"- **{key}**: {value}"
                for key, value in (insight_analysis.key_statistics.items() if insight_analysis else {})
            ]) + "\n\n## Patterns Identified\n" + "\n".join([
                f"- {pattern}"
                for pattern in (insight_analysis.patterns_identified[:3] if insight_analysis else [])
            ]),
            visualizations=insight_analysis.visualizations if insight_analysis else None
        )

        conclusions = ReportSection(
            title="Conclusions and Recommendations",
            content="## Recommendations\n" + "\n".join([
                f"{i+1}. {rec}"
                for i, rec in enumerate(critical_analysis.recommendations[:5])
            ]) + "\n\n## Future Implications\n" + "\n".join([
                f"- {imp}"
                for imp in (insight_analysis.future_implications[:3] if insight_analysis else [])
            ])
        )

        return ResearchReport(
            title=f"Research Report: {query}",
            executive_summary=critical_analysis.executive_summary,
            introduction=introduction,
            methodology=methodology,
            findings=findings,
            analysis=analysis,
            insights=insights,
            conclusions=conclusions,
            references=search_results.sources if search_results.sources else [],
            word_count=word_count,
            key_takeaways=key_takeaways[:5]
        )

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute report building for the workflow"""
        try:
            query = state["research_query"]
            search_results = state.get("search_results")
            critical_analysis = state.get("critical_analysis")
            insight_analysis = state.get("insight_analysis")

            if not search_results or not critical_analysis:
                return {
                    **state,
                    "current_step": "error",
                    "error": "Cannot generate report without search results and critical analysis"
                }

            report = self.generate_report(query, search_results, critical_analysis, insight_analysis)

            return {
                **state,
                "research_report": report,
                "current_step": "report_complete"
            }
        except Exception as e:
            return {
                **state,
                "current_step": "error",
                "error": f"Report generation failed: {str(e)}"
            }


class ResearchGraph:
    def __init__(self, api_key: str, tavily_api_key: str):
        self.api_key = api_key
        # Initialize agents
        self.retriever_agent = ContextualRetrieverAgent(api_key, tavily_api_key)
        self.analysis_agent = CriticalAnalysisAgent(api_key)
        self.insight_agent = InsightGenerationAgent(api_key)
        self.report_agent = ReportBuilderAgent(api_key)
        self.workflow = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(dict)

        # Add nodes for all four agents
        workflow.add_node("contextual_retriever", self.retriever_agent.execute)
        workflow.add_node("critical_analysis", self.analysis_agent.execute)
        workflow.add_node("insight_generation", self.insight_agent.execute)
        workflow.add_node("report_builder", self.report_agent.execute)

        # Set the entry point
        workflow.set_entry_point("contextual_retriever")

        # Add edges: sequential execution
        workflow.add_edge("contextual_retriever", "critical_analysis")
        workflow.add_edge("critical_analysis", "insight_generation")
        workflow.add_edge("insight_generation", "report_builder")

        # End after report
        workflow.add_edge("report_builder", END)

        return workflow.compile()

    def run(self, research_query: str) -> ResearchState:
        initial_state = {
            "research_query": research_query,
            "search_results": None,
            "critical_analysis": None,
            "insight_analysis": None,
            "research_report": None,
            "current_step": "initial",
            "error": None
        }

        result = self.workflow.invoke(initial_state)

        return ResearchState(**result)