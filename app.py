import streamlit as st
import os
from openai import OpenAI
import json
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
    st.markdown("## 📊 Research Results")

    if results.error:
        st.error(f"Error: {results.error}")
        return

    if results.search_results:
        search_res = results.search_results

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sources Found", search_res.total_results_found)
        with col2:
            st.metric("Top Sources Analyzed", len(search_res.sources))
        with col3:
            st.metric("Search Strategy", "Web Search")

        st.markdown("### 💡 Key Insights")
        for insight in search_res.key_insights:
            st.info(f"• {insight}")

        st.markdown("### 📚 Top 10 Sources by Relevance")

        for i, source in enumerate(search_res.sources, 1):
            with st.expander(f"{i}. {source.title} (Relevance: {source.relevance_score:.1f}%)"):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"**Type:** {source.source_type.value.replace('_', ' ').title()}")
                    st.markdown(f"**Domain:** {source.domain}")
                    if source.author:
                        st.markdown(f"**Author:** {source.author}")
                    if source.published_date:
                        st.markdown(f"**Published:** {source.published_date}")
                    st.markdown(f"**URL:** [{source.url}]({source.url})")

                with col2:
                    relevance_color = "green" if source.relevance_score >= 80 else "orange" if source.relevance_score >= 60 else "red"
                    st.markdown(f"### <span style='color: {relevance_color}'>{source.relevance_score:.0f}%</span>", unsafe_allow_html=True)
                    st.markdown("**Relevance Score**")

                st.markdown("**Summary:**")
                st.write(source.snippet)

                st.markdown("**Why Relevant:**")
                st.info(source.reasoning)

                st.markdown("---")

        with st.expander("📋 Export Results"):
            export_data = {
                "query": search_res.query,
                "timestamp": search_res.search_timestamp.isoformat(),
                "total_results": search_res.total_results_found,
                "sources": [
                    {
                        "title": s.title,
                        "url": s.url,
                        "relevance_score": s.relevance_score,
                        "type": s.source_type.value,
                        "snippet": s.snippet,
                        "reasoning": s.reasoning
                    }
                    for s in search_res.sources
                ]
            }
            st.download_button(
                label="Download JSON",
                data=json.dumps(export_data, indent=2),
                file_name=f"research_{search_res.search_timestamp.strftime('%Y%m%d_%H%M%S')}.json",
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
            with st.spinner("🔎 Searching and analyzing web sources..."):
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