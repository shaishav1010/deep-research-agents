import streamlit as st
import os
from openai import OpenAI
import json
import plotly.express as px
import plotly.graph_objects as go
from research_agent import ResearchGraph
from models import ResearchState

def init_session_state():
    if "openrouter_api_key" not in st.session_state:
        st.session_state.openrouter_api_key = ""
    if "tavily_api_key" not in st.session_state:
        st.session_state.tavily_api_key = ""
    if "api_keys_verified" not in st.session_state:
        st.session_state.api_keys_verified = False
    if "research_results" not in st.session_state:
        st.session_state.research_results = None
    if "research_in_progress" not in st.session_state:
        st.session_state.research_in_progress = False

def verify_openrouter_key(api_key):
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        response = client.models.list()
        return True
    except Exception as e:
        return False

def verify_tavily_key(api_key):
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        # Try a simple search to verify the key
        client.search("test", max_results=1)
        return True
    except Exception as e:
        return False

def display_research_results(results: ResearchState):
    st.markdown("## 🔬 Research Analysis Results")

    if results.error:
        st.error(f"Error: {results.error}")
        return

    # Show enhanced retriever summary with query refinement
    if results.search_results:
        with st.expander("📊 Contextual Retriever Summary", expanded=False):
            search_res = results.search_results

            # Query Refinement Section
            if search_res.refined_query:
                st.markdown("### 🎯 Query Interpretation & Refinement")
                st.info(f"**Interpretation:** {search_res.refined_query.interpretation}")

                st.markdown("**Subtopics Identified:**")
                for subtopic in search_res.refined_query.subtopics:
                    importance_bar = "🟢" * int(subtopic.importance * 5)
                    st.markdown(f"• **{subtopic.topic}** {importance_bar}")
                    st.markdown(f"  _{subtopic.description}_")

            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Sources", search_res.total_results_found)
            with col2:
                st.metric("Sources Analyzed", len(search_res.sources))
            with col3:
                if search_res.refined_query:
                    st.metric("Subtopics", len(search_res.refined_query.subtopics))
                else:
                    st.metric("Subtopics", "1")
            with col4:
                # Count source types
                source_types = set(s.source_type.value for s in search_res.sources)
                st.metric("Source Types", len(source_types))

            # Sources by Subtopic
            if search_res.sources_by_subtopic:
                st.markdown("### 📚 Sources by Subtopic")
                for topic, sources in search_res.sources_by_subtopic.items():
                    st.markdown(f"**{topic}:**")
                    for source in sources[:3]:
                        type_emoji = "📰" if source.source_type.value == "news_article" else \
                                    "🎓" if source.source_type.value == "academic_paper" else \
                                    "📊" if source.source_type.value == "report" else "🌐"
                        st.markdown(f"  {type_emoji} [{source.title[:60]}...]({source.url})")

            # Search Strategy
            st.markdown(f"**Search Strategy:** {search_res.search_strategy}")

    # Main display: Critical Analysis Results
    if results.critical_analysis:
        analysis = results.critical_analysis

        # Executive Summary
        st.markdown("### 📋 Executive Summary")
        st.info(analysis.executive_summary)

        # Key Findings
        st.markdown("### 🎯 Key Findings")
        for finding in analysis.key_findings:
            confidence_emoji = "🟢" if finding.confidence > 0.8 else "🟡" if finding.confidence > 0.5 else "🟠"
            with st.container():
                st.markdown(f"{confidence_emoji} **{finding.finding}**")
                st.markdown(f"   *Confidence: {finding.confidence:.0%} | Sources: {', '.join(finding.sources[:3])}*")

        # Contradictions and Conflicts
        if analysis.contradictions:
            st.markdown("### ⚠️ Contradictions Identified")
            for contradiction in analysis.contradictions:
                with st.container():
                    st.warning(f"""**Conflict between sources:**
• **{contradiction.source1}:** {contradiction.claim1}
• **{contradiction.source2}:** {contradiction.claim2}
*Explanation:* {contradiction.explanation}""")

        # Consensus Points
        st.markdown("### ✅ Consensus Points")
        for point in analysis.consensus_points:
            st.success(f"• {point}")

        # Source Validation
        st.markdown("### 🔍 Source Credibility")
        cols = st.columns(min(3, len(analysis.source_validations)))
        for i, validation in enumerate(analysis.source_validations[:3]):
            with cols[i]:
                color = "green" if validation.credibility_score >= 80 else "orange" if validation.credibility_score >= 60 else "red"
                st.markdown(f"**{validation.source_title[:50]}...**")
                st.markdown(f"<h3 style='color: {color}'>{validation.credibility_score:.0f}%</h3>", unsafe_allow_html=True)
                if validation.potential_biases:
                    st.markdown(f"⚠️ *Biases: {', '.join(validation.potential_biases)}*")

        # Knowledge Gaps
        if analysis.gaps_identified:
            st.markdown("### 🔓 Knowledge Gaps")
            for gap in analysis.gaps_identified:
                st.markdown(f"• {gap}")

        # Recommendations
        st.markdown("### 💡 Recommendations")
        for rec in analysis.recommendations:
            st.markdown(f"• **{rec}**")

        # Confidence Assessment
        st.markdown("### 📊 Overall Confidence")
        st.markdown(f"**{analysis.confidence_assessment}**")

    # Display Insight Analysis with Visualizations
    if results.insight_analysis:
        insights = results.insight_analysis

        st.markdown("---")
        st.markdown("## 📈 Statistical Insights & Visualizations")

        # Executive Insight Summary
        st.markdown("### 🎯 Insight Summary")
        st.info(insights.executive_insight_summary)

        # Key Statistics
        st.markdown("### 📊 Key Statistics")
        cols = st.columns(3)
        stats = insights.key_statistics

        with cols[0]:
            st.metric("Average Relevance", f"{stats.get('avg_relevance', 0):.1f}%")
            st.metric("Total Sources", stats.get('total_sources', 0))

        with cols[1]:
            st.metric("Max Relevance", f"{stats.get('max_relevance', 0):.0f}%")
            st.metric("Subtopics", stats.get('num_subtopics', 1))

        with cols[2]:
            st.metric("Min Relevance", f"{stats.get('min_relevance', 0):.0f}%")
            st.metric("Confidence", f"{stats.get('confidence_level', 0.5):.0%}")

        # Visualizations
        if insights.visualizations:
            st.markdown("### 📊 Data Visualizations")

            for viz in insights.visualizations:
                if viz.chart_type == "pie":
                    labels = viz.data.get("labels", [])
                    values = viz.data.get("values", [])

                    if labels and values:  # Only create chart if we have data
                        fig = go.Figure(data=[go.Pie(
                            labels=labels,
                            values=values,
                            hole=0.3
                        )])
                        fig.update_layout(
                            title=viz.title,
                            height=400,
                            showlegend=True
                        )
                        st.plotly_chart(fig, use_container_width=True)

                elif viz.chart_type == "bar":
                    x_data = viz.data.get("x", [])
                    y_data = viz.data.get("y", [])

                    if x_data and y_data:  # Only create chart if we have data
                        fig = go.Figure(data=[go.Bar(
                            x=x_data,
                            y=y_data,
                            marker_color='rgb(55, 83, 109)'
                        )])
                        fig.update_layout(
                            title=viz.title,
                            xaxis_title=viz.x_label if viz.x_label else "Categories",
                            yaxis_title=viz.y_label if viz.y_label else "Values",
                            height=400,
                            showlegend=False
                        )
                        fig.update_xaxes(tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)

                elif viz.chart_type == "line":
                    x_data = viz.data.get("x", [])
                    y_data = viz.data.get("y", [])

                    if x_data and y_data:  # Only create chart if we have data
                        fig = go.Figure(data=[go.Scatter(
                            x=x_data,
                            y=y_data,
                            mode='lines+markers',
                            line=dict(color='rgb(55, 83, 109)', width=2),
                            marker=dict(size=8)
                        )])
                        fig.update_layout(
                            title=viz.title,
                            xaxis_title=viz.x_label if viz.x_label else "X Axis",
                            yaxis_title=viz.y_label if viz.y_label else "Y Axis",
                            height=400,
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)

        # Source Categorization
        if insights.source_categorization:
            st.markdown("### 📂 Source Distribution")
            cols = st.columns(len(insights.source_categorization[:4]))
            for i, cat in enumerate(insights.source_categorization[:4]):
                with cols[i]:
                    st.markdown(f"**{cat.category}**")
                    st.progress(cat.percentage / 100)
                    st.markdown(f"{cat.count} sources ({cat.percentage:.1f}%)")

        # Statistical Insights
        if insights.statistical_insights:
            st.markdown("### 🔍 Statistical Findings")
            for insight in insights.statistical_insights:
                with st.expander(f"{insight.title} - {insight.insight_type.upper()}"):
                    st.markdown(f"**{insight.description}**")
                    st.markdown(f"*Significance: {insight.significance:.0%}*")
                    if insight.implications:
                        st.markdown("**Implications:**")
                        for impl in insight.implications:
                            st.markdown(f"• {impl}")

        # Patterns and Future Implications
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔮 Patterns Identified")
            for pattern in insights.patterns_identified:
                st.markdown(f"• {pattern}")

        with col2:
            st.markdown("### 🚀 Future Implications")
            for implication in insights.future_implications:
                st.markdown(f"• {implication}")

    # Export Options (moved here to include all data)
    if results.critical_analysis:
        analysis = results.critical_analysis
        # Export Options
        with st.expander("📋 Export Analysis"):
            export_data = {
                "query": results.research_query,
                "analysis_timestamp": analysis.analysis_timestamp.isoformat(),
                "executive_summary": analysis.executive_summary,
                "key_findings": [
                    {
                        "finding": f.finding,
                        "confidence": f.confidence,
                        "sources": f.sources
                    } for f in analysis.key_findings
                ],
                "contradictions": [
                    {
                        "source1": c.source1,
                        "source2": c.source2,
                        "claim1": c.claim1,
                        "claim2": c.claim2,
                        "explanation": c.explanation
                    } for c in analysis.contradictions
                ],
                "consensus_points": analysis.consensus_points,
                "gaps": analysis.gaps_identified,
                "recommendations": analysis.recommendations,
                "confidence": analysis.confidence_assessment
            }

            # Add insights data if available
            if results.insight_analysis:
                export_data["insights"] = {
                    "executive_summary": results.insight_analysis.executive_insight_summary,
                    "key_statistics": results.insight_analysis.key_statistics,
                    "patterns": results.insight_analysis.patterns_identified,
                    "future_implications": results.insight_analysis.future_implications,
                    "source_categorization": [
                        {
                            "category": cat.category,
                            "count": cat.count,
                            "percentage": cat.percentage
                        } for cat in results.insight_analysis.source_categorization
                    ],
                    "statistical_insights": [
                        {
                            "type": si.insight_type,
                            "title": si.title,
                            "description": si.description,
                            "significance": si.significance
                        } for si in results.insight_analysis.statistical_insights
                    ]
                }

            st.download_button(
                label="Download Complete Analysis JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"analysis_{analysis.analysis_timestamp.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

def main():
    st.set_page_config(
        page_title="AI Deep Research Agent",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    init_session_state()

    with st.sidebar:
        st.markdown("## 🔑 Configuration")

        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            value=st.session_state.openrouter_api_key,
            placeholder="sk-or-v1-...",
            help="Enter your OpenRouter API key. Get one at https://openrouter.ai/keys"
        )

        if api_key != st.session_state.openrouter_api_key:
            st.session_state.openrouter_api_key = api_key
            st.session_state.api_key_verified = False

        tavily_key = st.text_input(
            "Tavily API Key (Required)",
            type="password",
            value=st.session_state.tavily_api_key,
            placeholder="tvly-...",
            help="Required: Enter your Tavily API key for web search. Get one at https://tavily.com"
        )

        if tavily_key != st.session_state.tavily_api_key:
            st.session_state.tavily_api_key = tavily_key

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Verify Keys", type="primary", use_container_width=True):
                if not api_key or not tavily_key:
                    st.warning("Please enter both API keys")
                else:
                    with st.spinner("Verifying keys..."):
                        openrouter_valid = verify_openrouter_key(api_key)
                        tavily_valid = verify_tavily_key(tavily_key)

                        if openrouter_valid and tavily_valid:
                            st.session_state.api_keys_verified = True
                            st.success("✅ Both API keys verified!")
                        else:
                            st.session_state.api_keys_verified = False
                            if not openrouter_valid:
                                st.error("❌ Invalid OpenRouter API key")
                            if not tavily_valid:
                                st.error("❌ Invalid Tavily API key")

        with col2:
            if st.button("Clear Keys", use_container_width=True):
                st.session_state.openrouter_api_key = ""
                st.session_state.tavily_api_key = ""
                st.session_state.api_keys_verified = False
                st.rerun()

        if st.session_state.api_keys_verified:
            st.markdown("---")
            st.success("🟢 Ready to start research")

        st.markdown("---")
        st.markdown("### 📖 About")
        st.markdown("""
        This is a multi-agent AI deep researcher that helps you:
        - Conduct comprehensive research on any topic
        - Analyze and synthesize information
        - Generate detailed reports
        - Collaborate with AI agents
        """)

    st.markdown("# 🔬 AI Deep Research Agent")
    st.markdown("### Welcome to your intelligent research assistant")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        #### 🎯 Features
        - Multi-agent collaboration
        - Deep web research
        - Source verification
        - Report generation
        - Real-time analysis
        """)

    with col2:
        st.markdown("""
        #### 🚀 Getting Started
        1. Enter your OpenRouter API key
        2. Verify the connection
        3. Start your research
        4. Review generated reports
        """)

    with col3:
        st.markdown("""
        #### 💡 Use Cases
        - Academic research
        - Market analysis
        - Technical documentation
        - Competitive intelligence
        - Trend analysis
        """)

    st.markdown("---")

    if not st.session_state.api_keys_verified:
        st.info("👈 Please enter and verify both API keys in the sidebar to begin")

        col1, col2 = st.columns(2)

        with col1:
            with st.expander("🔗 How to get an OpenRouter API key"):
                st.markdown("""
                1. Visit [OpenRouter](https://openrouter.ai/)
                2. Sign up or log in
                3. Navigate to [API Keys](https://openrouter.ai/keys)
                4. Create a new API key
                5. Copy and paste it in the sidebar
                """)

        with col2:
            with st.expander("🔍 How to get a Tavily API key"):
                st.markdown("""
                1. Visit [Tavily](https://tavily.com)
                2. Sign up for a free account
                3. Navigate to API Keys section
                4. Create a new API key
                5. Copy and paste it in the sidebar
                """)
    else:
        st.success("✅ Both API keys verified! You're ready to start researching.")

        st.markdown("### 🎯 Start Your Research")

        research_topic = st.text_area(
            "What would you like to research today?",
            placeholder="Enter your research topic or question here...",
            height=100
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🔍 Start Research", type="primary", use_container_width=True):
                if research_topic:
                    st.session_state.research_in_progress = True
                    st.session_state.research_results = None
                else:
                    st.warning("Please enter a research topic")

        if st.session_state.research_in_progress:
            with st.spinner("🔎 Contextual Retriever Agent searching sources... → Critical Analysis Agent analyzing findings... → Insight Generation Agent creating visualizations..."):
                try:
                    research_graph = ResearchGraph(
                        st.session_state.openrouter_api_key,
                        st.session_state.tavily_api_key
                    )
                    result = research_graph.run(research_topic)
                    st.session_state.research_results = result
                    st.session_state.research_in_progress = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Research failed: {str(e)}")
                    st.session_state.research_in_progress = False

        if st.session_state.research_results:
            display_research_results(st.session_state.research_results)

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            Built with Streamlit • Powered by OpenRouter
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()