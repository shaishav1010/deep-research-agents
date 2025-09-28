# SynthiVerseAI - Deep Research Multi-Agent System

**Welcome to advanced synthesis and deep exploration**

SynthiVerseAI is an AI-powered deep research assistant that leverages multiple specialized agents to conduct comprehensive, multi-source investigations. Built with LangChain, LangGraph, and OpenRouter, it orchestrates a sophisticated pipeline of AI agents to deliver thorough research reports with critical analysis, data visualizations, and actionable insights.

## 🚀 Features

- **Multi-Agent Architecture**: Four specialized agents working in sequence to deliver comprehensive research
- **Real-time Progress Tracking**: Live status updates showing which agent is currently processing
- **Advanced Web Search**: Powered by Tavily API for high-quality web research
- **Data Visualization**: Automatic generation of charts and graphs from analyzed data
- **Multiple Export Formats**: Export reports as PDF, Word (DOCX), or XML
- **Structured Output**: Uses Pydantic models for consistent, validated data structures

## 🏗️ Architecture

### Agent Pipeline

The system uses **LangGraph** to orchestrate a sequential workflow of four specialized agents:

```
┌─────────────────────┐
│ Contextual Retriever│
│      Agent          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Critical Analysis  │
│      Agent          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Insight Generation  │
│      Agent          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Report Builder    │
│      Agent          │
└─────────────────────┘
```

### Agent Descriptions

#### 1. 🔍 Contextual Retriever Agent
**Purpose**: Query refinement and comprehensive information gathering

- **Refines user queries** into optimized search terms
- **Identifies relevant subtopics** to ensure comprehensive coverage
- **Performs multi-source searches** using Tavily API
- **Aggregates diverse perspectives** from various source types (academic, news, reports, websites)

**Key Technologies**:
- Tavily API for web search
- LangChain's structured output parsing
- Query optimization techniques

#### 2. 🧠 Critical Analysis Agent
**Purpose**: Synthesize findings and perform deep analysis

- **Validates source credibility** with scoring system
- **Identifies contradictions** between different sources
- **Extracts key findings** with confidence levels
- **Builds consensus points** from multiple perspectives
- **Detects potential biases** in sources

**Output**: Structured analysis with confidence scores, validated sources, and identified patterns

#### 3. 📊 Insight Generation Agent
**Purpose**: Transform analyzed data into visual insights

- **Creates data visualizations** from critical analysis findings
- **Generates charts** including:
  - Confidence levels of key findings
  - Source credibility scores
  - Contradictions vs consensus analysis
  - Finding distribution patterns
- **Extracts patterns and trends** from the analyzed data
- **Provides visual storytelling** to complement textual analysis

**Note**: Visualizations are based on the critical analysis output, not raw search results

#### 4. 📝 Report Builder Agent
**Purpose**: Compile comprehensive, professional research reports

- **Structures findings** into professional report format
- **Generates executive summary** with key takeaways
- **Creates detailed sections**:
  - Introduction
  - Methodology
  - Findings
  - Analysis
  - Insights
  - Conclusions
- **Includes proper citations** and references
- **Calculates word count** for the report

## 🔧 How LangChain and LangGraph Work Together

### LangChain's Role
LangChain provides the foundational components:

1. **LLM Integration**:
   - Connects to OpenRouter API (using `ChatOpenAI` with custom base URL)
   - Manages prompts and completions with `ChatPromptTemplate`

2. **Structured Output**:
   - `PydanticOutputParser` ensures LLM responses match defined schemas
   - Validates and parses agent outputs into Pydantic models

3. **Tools and Utilities**:
   - Integration with Tavily for web search
   - Document processing and text manipulation

### LangGraph's Orchestration
LangGraph manages the multi-agent workflow:

1. **State Management**:
   ```python
   workflow = StateGraph(dict)
   ```
   - Maintains shared state between agents
   - Passes outputs from one agent as inputs to the next

2. **Node Definition**:
   ```python
   workflow.add_node("contextual_retriever", self.retriever_agent.execute)
   workflow.add_node("critical_analysis", self.analysis_agent.execute)
   workflow.add_node("insight_generation", self.insight_agent.execute)
   workflow.add_node("report_builder", self.report_agent.execute)
   ```
   - Each agent is a node in the graph
   - Nodes execute their specialized tasks

3. **Edge Connections**:
   ```python
   workflow.add_edge("contextual_retriever", "critical_analysis")
   workflow.add_edge("critical_analysis", "insight_generation")
   workflow.add_edge("insight_generation", "report_builder")
   workflow.add_edge("report_builder", END)
   ```
   - Defines the sequential flow
   - Ensures data flows correctly between agents

4. **Compilation and Execution**:
   ```python
   compiled_graph = workflow.compile()
   result = compiled_graph.invoke({"query": user_query})
   ```
   - Compiles the graph for execution
   - Manages the entire pipeline from query to report

### Data Flow

1. **User Query** → Contextual Retriever Agent
2. **Search Results** → Critical Analysis Agent
3. **Critical Analysis** → Insight Generation Agent
4. **Insights + Analysis** → Report Builder Agent
5. **Final Report** → User (with export options)

## 📋 Prerequisites

- Python 3.8+
- OpenRouter API Key
- Tavily API Key

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/deep-research-agents.git
cd deep-research-agents
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory:
```env
OPENROUTER_API_KEY=your_openrouter_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## 🚦 Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Enter your API keys in the sidebar (if not set via environment variables)

4. Input your research query

5. Watch as the agents process your request in real-time

6. Export your report in your preferred format

## 📁 Project Structure

```
deep-research-agents/
│
├── app.py                 # Streamlit UI and main application
├── research_agent.py      # Core agent implementations
├── models.py             # Pydantic models for data structures
├── export_utils.py       # PDF, Word, XML export functionality
├── output_capture.py     # Real-time console output capture
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

### Key Files

- **`research_agent.py`**: Contains all four agent classes and the `ResearchOrchestrator` that manages the LangGraph workflow
- **`models.py`**: Defines Pydantic models for structured data (search results, analysis, reports)
- **`app.py`**: Streamlit interface with real-time progress tracking and export functionality
- **`export_utils.py`**: Handles document generation in multiple formats

## 🔑 Key Technologies

- **LangChain**: Provides LLM abstractions, prompt management, and output parsing
- **LangGraph**: Orchestrates the multi-agent workflow with state management
- **OpenRouter**: Access to various LLMs (default: GPT-4o-mini)
- **Tavily API**: High-quality web search and content retrieval
- **Streamlit**: Interactive web interface
- **Pydantic**: Data validation and structured outputs
- **Plotly**: Interactive data visualizations
- **ReportLab**: PDF generation
- **python-docx**: Word document creation

## 🎯 Use Cases

- Academic research and literature reviews
- Market analysis and competitive intelligence
- Technology trend analysis
- Policy research and analysis
- Medical and scientific research synthesis
- Business intelligence gathering
- Due diligence investigations

## 🔄 Workflow Example

1. **User Query**: "What are the latest developments in quantum computing?"

2. **Contextual Retriever**:
   - Refines to search for "quantum computing breakthroughs 2024"
   - Identifies subtopics: quantum supremacy, error correction, commercial applications
   - Searches across academic papers, news, and industry reports

3. **Critical Analysis**:
   - Validates sources (IBM Research: 95%, Tech Blog: 60%)
   - Identifies consensus: Error correction is the main challenge
   - Finds contradiction: Timeline for commercial viability varies

4. **Insight Generation**:
   - Creates visualization of confidence levels for findings
   - Generates source credibility chart
   - Shows distribution of research focus areas

5. **Report Builder**:
   - Compiles 2000+ word comprehensive report
   - Includes executive summary with 5 key takeaways
   - Properly cites all sources

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenRouter for LLM access
- Tavily for search API
- LangChain and LangGraph teams for the excellent frameworks
- Streamlit for the intuitive web framework

---

**Built with ❤️ for deep research and knowledge synthesis**