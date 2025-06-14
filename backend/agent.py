import os
from typing import TypedDict, Any, List, Dict, Optional, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from rich.console import Console
from dotenv import load_dotenv
import operator
import asyncio
from concurrent.futures import ThreadPoolExecutor

from tools.web_search_tool import search_tavily
from tools.vector_store_search_tool import VectorStoreSearchTool

console = Console()
load_dotenv(override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=GOOGLE_API_KEY,
    temperature=0.3,
)


class AgentState(TypedDict):
    """Streamlined state definition for the VC Pitch Assistant workflow."""
    query: str
    original_query: str
    query_type: str  # 'pitch_deck', 'funding', 'investor', 'market', 'general'
    search_results: Dict[str, List[Dict[str, Any]]]
    final_answer: str
    conversation_history: List[Dict[str, str]]
    error_message: Optional[str]
    search_metadata: Dict[str, Any]


class VCPitchAssistantAgent:
    def __init__(self):
        """Initialize the streamlined VC Pitch Assistant agent."""
        self.llm = llm
        self.console = console
        
        # Initialize tools with error handling
        self._initialize_tools()
        self._setup_enhanced_prompts()
        self.workflow = self._build_streamlined_workflow()

    def _initialize_tools(self):
        """Initialize search tools with proper error handling."""
        try:
            self.vector_search_tool = VectorStoreSearchTool()
            console.print("✅ Vector search tool initialized", style="green")
        except Exception as e:
            console.print(f"❌ Vector search tool error: {e}", style="red")
            self.vector_search_tool = None

    def _setup_enhanced_prompts(self):
        """Setup enhanced prompt templates with better structure and context awareness."""
        
        # Query classification prompt
        self.query_classifier_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
Classify this VC/startup query into ONE primary category:

Query: "{query}"

Categories:
- pitch_deck: Building, reviewing, or improving pitch decks
- funding: Fundraising strategies, timing, or processes  
- investor: Targeting investors, investor relations, or networking
- market: Market analysis, competitive landscape, or positioning
- metrics: KPIs, financial projections, or performance tracking
- general: Other startup/VC related questions

Respond with ONLY the category name (e.g., "pitch_deck").
"""
        )

        # Enhanced answer generation prompt
        self.answer_generation_prompt = PromptTemplate(
            input_variables=["query", "query_type", "pitch_deck_context", "funding_news_context", "conversation_history"],
            template="""
You are an expert VC Pitch Assistant specializing in startup fundraising and pitch deck optimization.

USER QUERY: {query}
QUERY TYPE: {query_type}

PITCH DECK DATABASE INSIGHTS:
{pitch_deck_context}

CURRENT FUNDING LANDSCAPE:
{funding_news_context}

CONVERSATION CONTEXT:
{conversation_history}

RESPONSE GUIDELINES:

For PITCH_DECK queries:
• Provide specific slide-by-slide recommendations
• Reference successful examples from the database when available
• Include design and content best practices
• Suggest metrics and data points to highlight

For FUNDING queries:
• Outline fundraising timeline and milestones
• Recommend funding stages and amounts based on traction
• Provide current market conditions context
• Include investor outreach strategies

For INVESTOR queries:
• Suggest specific investor types matching the startup profile
• Provide networking and approach strategies
• Include due diligence preparation tips
• Recommend warm introduction tactics

For MARKET queries:
• Deliver competitive analysis frameworks
• Provide market sizing methodologies
• Include positioning and differentiation strategies

For METRICS queries:
• Specify KPIs relevant to the business model
• Provide benchmark data when available
• Include financial projection templates
• Suggest tracking and reporting methods

RESPONSE FORMAT:
1. **Executive Summary**: 2-3 sentence overview
2. **Key Recommendations**: 3-5 specific, actionable items
3. **Supporting Evidence**: Examples from database/market
4. **Next Steps**: Clear action items
5. **Additional Resources**: Relevant follow-up areas

Keep responses comprehensive yet concise. Always ground recommendations in real examples and current market data.

RESPONSE:
"""
        )

    def query_classification_node(self, state: AgentState) -> Dict[str, Any]:
        """Classify the query type for targeted processing."""
        try:
            console.print("🔍 Classifying query type...", style="blue")
            
            chain = self.query_classifier_prompt | self.llm
            query_type = chain.invoke({"query": state["query"]}).strip().lower()
            
            # Enhance query based on type
            enhanced_query = self._enhance_query(state["query"], query_type)
            
            console.print(f"✅ Query classified as: {query_type}", style="green")
            
            return {
                "query_type": query_type,
                "query": enhanced_query,
                "original_query": state["query"]
            }
        except Exception as e:
            console.print(f"❌ Query classification failed: {e}", style="red")
            return {
                "query_type": "general",
                "query": state["query"],
                "original_query": state["query"]
            }

    def _enhance_query(self, original_query: str, query_type: str) -> str:
        """Enhance query based on classification for better search results."""
        enhancements = {
            "pitch_deck": f"{original_query} pitch deck slides template structure",
            "funding": f"{original_query} startup fundraising venture capital series A seed",
            "investor": f"{original_query} VC investors funding startup investment",
            "market": f"{original_query} market analysis competitive landscape startup",
            "metrics": f"{original_query} startup KPI metrics financial projections"
        }
        return enhancements.get(query_type, original_query)

    def parallel_search_node(self, state: AgentState) -> Dict[str, Any]:
        """Perform parallel search across vector store and web with optimized queries."""
        try:
            console.print("🔍 Executing parallel search...", style="blue")
            
            search_results = {"vector": [], "web": []}
            search_metadata = {"vector_count": 0, "web_count": 0, "search_time": 0}
            
            import time
            start_time = time.time()
            
            # Use ThreadPoolExecutor for parallel execution
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both searches
                vector_future = executor.submit(self._vector_search, state["query"], state["query_type"])
                web_future = executor.submit(self._web_search, state["query"], state["query_type"])
                
                # Get results
                vector_results = vector_future.result()
                web_results = web_future.result()
                
                search_results["vector"] = vector_results
                search_results["web"] = web_results
            
            search_metadata.update({
                "vector_count": len(vector_results),
                "web_count": len(web_results),
                "search_time": time.time() - start_time
            })
            
            console.print(
                f"✅ Search completed: {search_metadata['vector_count']} pitch decks, "
                f"{search_metadata['web_count']} news articles in {search_metadata['search_time']:.2f}s", 
                style="green"
            )
            
            return {
                "search_results": search_results,
                "search_metadata": search_metadata
            }
            
        except Exception as e:
            error_msg = f"Parallel search failed: {str(e)}"
            console.print(f"❌ {error_msg}", style="red")
            return {
                "search_results": {"vector": [], "web": []},
                "search_metadata": {"error": error_msg},
                "error_message": error_msg
            }

    def _vector_search(self, query: str, query_type: str) -> List[Dict[str, Any]]:
        """Optimized vector search with query-type specific parameters."""
        if not self.vector_search_tool:
            return []
        
        try:
            # Adjust search parameters based on query type
            k_values = {
                "pitch_deck": 6,
                "funding": 4,
                "investor": 3,
                "market": 5,
                "metrics": 4,
                "general": 5
            }
            
            k = k_values.get(query_type, 5)
            documents = self.vector_search_tool.search(query, k=k)
            return self.vector_search_tool.format_results(documents, max_content_length=400)
            
        except Exception as e:
            console.print(f"Vector search error: {e}", style="yellow")
            return []

    def _web_search(self, query: str, query_type: str) -> List[Dict[str, Any]]:
        """Optimized web search with query-type specific enhancement."""
        try:
            # Create targeted web search queries
            web_queries = {
                "pitch_deck": f"{query} pitch deck template 2024 2025",
                "funding": f"{query} startup funding rounds 2024 2025 venture capital",
                "investor": f"{query} VC investors startup funding 2024 2025",
                "market": f"{query} market trends startup industry 2024 2025",
                "metrics": f"{query} startup metrics benchmarks KPI 2024 2025"
            }
            
            enhanced_query = web_queries.get(query_type, f"{query} startup 2024 2025")
            
            search_results = search_tavily(
                query=enhanced_query,
                max_res=6,
                search_depth="advanced"
            )
            
            if search_results.get("error"):
                return []
                
            return search_results.get("results", [])
            
        except Exception as e:
            console.print(f"Web search error: {e}", style="yellow")
            return []

    def generate_answer_node(self, state: AgentState) -> Dict[str, Any]:
        """Generate enhanced answer using structured context."""
        try:
            console.print("🤖 Generating specialized response...", style="blue")
            
            # Prepare structured context
            pitch_deck_context = self._format_pitch_deck_context(
                state["search_results"]["vector"]
            )
            funding_news_context = self._format_funding_news_context(
                state["search_results"]["web"]
            )
            conversation_history = self._format_conversation_history(
                state.get("conversation_history", [])
            )
            
            # Generate response
            chain = self.answer_generation_prompt | self.llm
            response = chain.invoke({
                "query": state["original_query"],
                "query_type": state["query_type"],
                "pitch_deck_context": pitch_deck_context,
                "funding_news_context": funding_news_context,
                "conversation_history": conversation_history
            })
            
            console.print("✅ Response generated successfully", style="green")
            
            return {
                "final_answer": response,
                "error_message": None
            }
            
        except Exception as e:
            error_msg = f"Answer generation failed: {str(e)}"
            console.print(f"❌ {error_msg}", style="red")
            return {
                "final_answer": f"I apologize, but I encountered an error while generating your response: {error_msg}",
                "error_message": error_msg
            }

    def _format_pitch_deck_context(self, vector_results: List[Dict[str, Any]]) -> str:
        """Format pitch deck database results with better structure."""
        if not vector_results:
            return "No relevant pitch deck examples found in database."
        
        context_parts = []
        for i, result in enumerate(vector_results[:4], 1):
            content = result.get("content", "")
            metadata = result.get("metadata", {})
            
            startup_name = metadata.get("startup_name", "Unknown Startup")
            stage = metadata.get("funding_stage", "Unknown Stage")
            industry = metadata.get("industry", "Unknown Industry")
            
            context_parts.append(
                f"EXAMPLE {i}: {startup_name} ({stage} - {industry})\n"
                f"Key Insights: {content[:300]}...\n"
            )
        
        return "\n".join(context_parts)

    def _format_funding_news_context(self, web_results: List[Dict[str, Any]]) -> str:
        """Format current funding news with relevance filtering."""
        if not web_results:
            return "No current funding news available."
        
        context_parts = []
        for i, result in enumerate(web_results[:4], 1):
            title = result.get("title", "Unknown Title")
            content = result.get("content", "")
            
            context_parts.append(
                f"NEWS {i}: {title}\n"
                f"Summary: {content[:250]}...\n"
            )
        
        return "\n".join(context_parts)

    def _format_conversation_history(self, history: List[Dict[str, str]]) -> str:
        """Format conversation history for better context."""
        if not history:
            return "First interaction in this session."
        
        formatted = []
        for entry in history[-2:]:  # Last 2 exchanges for context
            query = entry.get("query", "")
            answer = entry.get("answer", "")
            formatted.append(f"Previous Q: {query[:100]}...\nPrevious A: {answer[:200]}...")
        
        return "\n\n".join(formatted)

    def _build_streamlined_workflow(self) -> StateGraph:
        """Build optimized LangGraph workflow with better flow control."""
        workflow = StateGraph(AgentState)
        
        # Add nodes in logical sequence
        workflow.add_node("classify_query", self.query_classification_node)
        workflow.add_node("parallel_search", self.parallel_search_node)
        workflow.add_node("generate_answer", self.generate_answer_node)
        
        # Create linear flow for better control
        workflow.add_edge(START, "classify_query")
        workflow.add_edge("classify_query", "parallel_search")
        workflow.add_edge("parallel_search", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        # Compile with memory
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def run(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Execute the streamlined VC Pitch Assistant workflow."""
        try:
            console.print(f"🚀 Processing: '{query}'", style="bold blue")
            
            # Prepare initial state
            initial_state = {
                "query": query,
                "original_query": query,
                "query_type": "",
                "search_results": {"vector": [], "web": []},
                "final_answer": "",
                "conversation_history": conversation_history or [],
                "error_message": None,
                "search_metadata": {}
            }
            
            # Execute workflow
            config = {"configurable": {"thread_id": f"session_{hash(query)}"}}
            final_state = self.workflow.invoke(initial_state, config)
            
            # Update conversation history
            if conversation_history is not None:
                conversation_history.append({
                    "query": query,
                    "answer": final_state.get("final_answer", "")[:500]  # Truncate for memory
                })
            
            # Prepare response
            search_meta = final_state.get("search_metadata", {})
            
            console.print("🎉 Response ready!", style="bold green")
            
            return {
                "answer": final_state.get("final_answer", "Unable to generate response."),
                "query_type": final_state.get("query_type", "unknown"),
                "pitch_deck_count": search_meta.get("vector_count", 0),
                "news_count": search_meta.get("web_count", 0),
                "processing_time": search_meta.get("search_time", 0),
                "error": final_state.get("error_message")
            }
            
        except Exception as e:
            error_msg = f"Workflow execution failed: {str(e)}"
            console.print(f"💥 {error_msg}", style="bold red")
            return {
                "answer": f"I apologize, but I encountered an error: {error_msg}",
                "query_type": "error",
                "pitch_deck_count": 0,
                "news_count": 0,
                "processing_time": 0,
                "error": error_msg
            }


def main():
    """Demonstration of the streamlined VC Pitch Assistant."""
    try:
        # Initialize agent
        agent = VCPitchAssistantAgent()
        
        # Test queries covering different categories
        test_queries = [
            "How should I structure my Series A pitch deck for a fintech startup?",
            "What are the current funding trends for AI startups in 2024?",
            "Which investors should I target for climate tech seed funding?",
            "What metrics should I track for a B2B SaaS startup?",
            "How do I position against competitors in the e-commerce space?"
        ]
        
        conversation_history = []
        
        for query in test_queries:
            console.print(f"\n{'='*80}", style="bold")
            
            result = agent.run(query, conversation_history)
            
            console.print(f"\nQuery: {query}", style="bold cyan")
            console.print(f"Type: {result['query_type']} | Time: {result['processing_time']:.2f}s", style="yellow")
            console.print(f"Sources: {result['pitch_deck_count']} pitch decks, {result['news_count']} news", style="yellow")
            console.print(f"\n📊 Response:", style="bold green")
            console.print(result["answer"], style="white")
            
            if result["error"]:
                console.print(f"\nError: {result['error']}", style="red")
            
            # Brief pause between queries
            import time
            time.sleep(0.5)
    
    except Exception as e:
        console.print(f"Failed to run demo: {e}", style="bold red")


if __name__ == "__main__":
    main()