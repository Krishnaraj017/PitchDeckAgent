import streamlit as st
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid
from agent import VCPitchAssistantAgent
# Configure Streamlit page
st.set_page_config(
    page_title="VC Pitch Assistant with Guardrails",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables."""
    if 'agent' not in st.session_state:
        try:
            st.session_state.agent = VCPitchAssistantAgent()
            st.session_state.initialized = True
            st.session_state.init_error = None
        except Exception as e:
            st.session_state.initialized = False
            st.session_state.init_error = str(e)

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]

    if 'audit_logs' not in st.session_state:
        st.session_state.audit_logs = []
    
    if 'processing' not in st.session_state:
        st.session_state.processing = False

# Initialize session state
initialize_session_state()

# Custom CSS for better styling
st.markdown("""
<style>
    .stAlert > div {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        background-color: #f8f9fa;
    }
    
    .user-message {
        border-left-color: #17a2b8;
        background-color: #e1f5fe;
    }
    
    .assistant-message {
        border-left-color: #28a745;
        background-color: #f1f8e9;
    }
    
    .error-message {
        border-left-color: #dc3545;
        background-color: #f8d7da;
    }
</style>
""", unsafe_allow_html=True)

def display_safety_status(risk_level: str, violations: List[str], safety_summary: str):
    """Display safety status with appropriate styling."""
    if risk_level == "critical":
        st.error(f"🚨 **Critical Risk**: {safety_summary}")
    elif risk_level == "high":
        st.warning(f"⚠️ **High Risk**: {safety_summary}")
    elif risk_level == "medium":
        st.warning(f"🔶 **Medium Risk**: {safety_summary}")
    else:
        st.success(f"✅ **Low Risk**: {safety_summary}")
    
    if violations:
        st.warning(f"**Guardrail Violations**: {', '.join(violations)}")

def display_processing_metrics(result: Dict[str, Any]):
    """Display processing metrics in a nice format."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Processing Time", 
            f"{result.get('processing_time', 0):.2f}s",
            help="Time taken to process the query"
        )
    
    with col2:
        st.metric(
            "Pitch Decks", 
            result.get('pitch_deck_count', 0),
            help="Number of pitch deck examples found"
        )
    
    with col3:
        st.metric(
            "News Articles", 
            result.get('news_count', 0),
            help="Number of current news articles found"
        )
    
    with col4:
        st.metric(
            "Safety Checks", 
            result.get('processing_metadata', {}).get('total_safety_checks', 0),
            help="Total number of safety checks performed"
        )

def display_chat_message(message: Dict[str, Any], is_user: bool = False):
    """Display a chat message with proper styling."""
    if is_user:
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 You:</strong><br>
            {message.get('content', message.get('query', ''))}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>🤖 VC Assistant:</strong><br>
            {message.get('content', message.get('answer', ''))}
        </div>
        """, unsafe_allow_html=True)

def process_query(query: str):
    """Process the user query with the agent."""
    try:
        # Set processing flag
        st.session_state.processing = True
        
        # Add user message to chat
        user_message = {
            'type': 'user',
            'content': query,
            'timestamp': datetime.now().isoformat()
        }
        st.session_state.chat_history.append(user_message)
        
        # Process query with the enhanced agent
        start_time = time.time()
        
        # Debug: Print to check if agent exists
        st.write(f"Debug: Agent initialized: {st.session_state.initialized}")
        st.write(f"Debug: Agent object: {type(st.session_state.agent)}")
        
        result = st.session_state.agent.run_enhanced(
            query,
            session_id=st.session_state.session_id,
            conversation_history=st.session_state.conversation_history
        )
        
        processing_time = time.time() - start_time
        result['actual_processing_time'] = processing_time
        
        # Add assistant response to chat
        assistant_message = {
            'type': 'assistant',
            'content': result.get('answer', 'No response generated'),
            'timestamp': datetime.now().isoformat(),
            'metadata': result
        }
        st.session_state.chat_history.append(assistant_message)
        
        # Update conversation history for context
        st.session_state.conversation_history.append({
            'query': query,
            'response': result.get('answer', ''),
            'timestamp': datetime.now().isoformat()
        })
        
        # Update audit logs
        st.session_state.audit_logs.append({
            'query': query,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
        return True, None
        
    except Exception as e:
        error_message = {
            'type': 'assistant',
            'content': f"❌ Error processing your request: {str(e)}",
            'timestamp': datetime.now().isoformat(),
            'metadata': {'error': str(e), 'risk_level': 'critical'}
        }
        st.session_state.chat_history.append(error_message)
        return False, str(e)
    
    finally:
        # Reset processing flag
        st.session_state.processing = False

def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🚀 VC Pitch Assistant with Enhanced Guardrails")
    st.markdown("**Secure, Compliant, and Human-Supervised AI Assistant for Venture Capital**")
    
    # Check initialization
    if not st.session_state.get('initialized', False):
        st.error("❌ Failed to initialize agent")
        if st.session_state.get('init_error'):
            st.error(f"Error details: {st.session_state.init_error}")
        st.info("Please check your environment variables and dependencies.")
        
        # Show retry button
        if st.button("🔄 Retry Initialization"):
            # Clear the session state and try again
            for key in ['agent', 'initialized', 'init_error']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        return
    
    # Sidebar for controls and settings
    with st.sidebar:
        st.header("🛠️ Control Panel")
        
        # Session info
        st.subheader("Session Information")
        st.code(f"Session ID: {st.session_state.session_id}")
        st.metric("Messages", len(st.session_state.chat_history))
        
        # Safety settings
        st.subheader("🛡️ Safety Settings")
        auto_approval = st.toggle("Auto-approve low-risk responses", value=True)
        show_audit_trail = st.toggle("Show audit trail", value=False)
        show_debug_info = st.toggle("Show debug information", value=False)
        
        # Quick test scenarios
        st.subheader("🧪 Quick Test Scenarios")
        
        test_scenarios = {
            "Low Risk - Market Research": "What are the current funding trends for AI startups in 2024?",
            "Medium Risk - Investor Strategy": "How can I find the best VCs for my fintech startup?",
            "Edge Case - Ambiguous Query": "Help me with my pitch",
        }
        
        for scenario_name, query in test_scenarios.items():
            if st.button(f"Test: {scenario_name}", key=f"test_{scenario_name}"):
                if not st.session_state.processing:
                    with st.spinner("Processing test query..."):
                        success, error = process_query(query)
                    if success:
                        st.success("Test query processed!")
                    else:
                        st.error(f"Test failed: {error}")
                    st.rerun()
        
        # Clear history
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.conversation_history = []
            st.session_state.audit_logs = []
            st.rerun()
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat Interface")
        # Display chat history
        if st.session_state.chat_history:
            for i, message in enumerate(st.session_state.chat_history):
            # Apply custom styling with color: black
                st.markdown("""
                <style>
                .chat-message {
                    color: black !important;
                }
                </style>
                """, unsafe_allow_html=True)
                
                if message['type'] == 'user':
                    display_chat_message(message, is_user=True)
                else:
                    display_chat_message(message, is_user=False)
                    
                    # Show additional metadata for assistant messages
                    if show_debug_info and 'metadata' in message:
                        with st.expander(f"Debug Info - Message {i+1}"):
                            st.json(message['metadata'])
        else:
            st.info("👋 Welcome! Ask me anything about venture capital, pitch decks, or startup funding.")
        
        # Input area
        st.subheader("💭 Ask Your Question")
        
        # Use a form to handle the input properly
        with st.form(key="query_form", clear_on_submit=True):
            query_input = st.text_area(
                "Enter your VC question:",
                height=100,
                help="Enter your venture capital related question. The system will apply safety checks and guardrails.",
                placeholder="e.g., What are the key elements of a successful pitch deck?"
            )
            
            # Submit button
            submitted = st.form_submit_button(
                "🚀 Send Message", 
                type="primary",
                disabled=st.session_state.processing
            )
            
            # Process the query when submitted
            if submitted and query_input.strip():
                if not st.session_state.processing:
                    with st.spinner("🤔 Processing your query with safety checks..."):
                        success, error = process_query(query_input.strip())
                    
                    if success:
                        st.success("✅ Query processed successfully!")
                    else:
                        st.error(f"❌ Processing failed: {error}")
                    
                    # Rerun to show the new messages
                    time.sleep(0.5)  # Brief pause to show the success message
                    st.rerun()
                elif query_input.strip():
                    st.warning("⏳ Please wait for the current query to finish processing.")
    
    with col2:
        st.subheader("📊 Live Metrics")
        
        # Processing status
        if st.session_state.processing:
            st.info("🔄 Processing query...")
        else:
            st.success("✅ Ready for queries")
        
        # Real-time metrics display
        if st.session_state.chat_history:
            last_response = None
            for msg in reversed(st.session_state.chat_history):
                if msg['type'] == 'assistant' and 'metadata' in msg:
                    last_response = msg['metadata']
                    break
            
            if last_response:
                st.subheader("🛡️ Last Response Safety")
                display_safety_status(
                    last_response.get('risk_level', 'unknown'),
                    last_response.get('guardrail_violations', []),
                    last_response.get('safety_summary', 'No safety info available')
                )
                
                st.subheader("⚡ Performance Metrics")
                display_processing_metrics(last_response)
        
        # System status
        st.subheader("🔧 System Status")
        st.success("✅ Agent Online")
        st.info(f"💬 {len(st.session_state.chat_history)} Messages")
        st.info(f"📋 {len(st.session_state.audit_logs)} Audit Events")
    
    # Debug information at the bottom (optional)
    if show_debug_info:
        with st.expander("🐛 Debug Information"):
            st.write("**Session State Keys:**", list(st.session_state.keys()))
            st.write("**Agent Initialized:**", st.session_state.get('initialized', False))
            st.write("**Processing:**", st.session_state.get('processing', False))
            if st.session_state.get('init_error'):
                st.write("**Init Error:**", st.session_state.init_error)

if __name__ == "__main__":
    main()