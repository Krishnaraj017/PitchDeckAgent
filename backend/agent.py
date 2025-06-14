import os
import json
import hashlib
from typing import TypedDict, Any, List, Dict, Optional, Annotated, Literal, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import logging

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from dotenv import load_dotenv

from tools.web_search_tool import search_tavily
from tools.vector_store_search_tool import VectorStoreSearchTool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console()
load_dotenv(override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=GOOGLE_API_KEY,
    temperature=0.3,
)

# =============================================================================
# GUARDRAILS AND SAFETY ENUMS/CLASSES
# =============================================================================


class RiskLevel(Enum):
    """Risk levels for content and actions."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionType(Enum):
    """Types of actions the agent can perform."""

    SEARCH = "search"
    GENERATE = "generate"
    RECOMMEND = "recommend"
    ANALYZE = "analyze"
    EXTERNAL_CALL = "external_call"


class GuardrailViolation(Enum):
    """Types of guardrail violations."""

    INAPPROPRIATE_CONTENT = "inappropriate_content"
    DATA_PRIVACY = "data_privacy"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    RATE_LIMIT = "rate_limit"
    CONTENT_POLICY = "content_policy"
    HALLUCINATION_RISK = "hallucination_risk"
    BIAS_DETECTED = "bias_detected"


@dataclass
class SafetyCheck:
    """Safety check result."""

    passed: bool
    risk_level: RiskLevel
    violations: List[GuardrailViolation] = field(default_factory=list)
    confidence: float = 1.0
    explanation: str = ""
    auto_approved: bool = False


@dataclass
class HumanApproval:
    """Human approval request and response."""

    request_id: str
    action_type: ActionType
    content: str
    risk_assessment: SafetyCheck
    timestamp: datetime
    approved: Optional[bool] = None
    human_feedback: str = ""
    timeout_seconds: int = 300


# =============================================================================
# AGENT STATE
# =============================================================================


class AgentState(TypedDict):
    """Enhanced state with guardrails and human-in-loop support."""

    # Core workflow state
    query: str
    original_query: str
    query_type: str
    search_results: Dict[str, List[Dict[str, Any]]]
    final_answer: str
    conversation_history: List[Dict[str, str]]
    error_message: Optional[str]
    search_metadata: Dict[str, Any]

    # Safety and guardrails
    safety_checks: List[SafetyCheck]
    pending_approvals: List[HumanApproval]
    risk_level: str
    guardrail_violations: List[str]

    # Audit and compliance
    audit_trail: List[Dict[str, Any]]
    content_filters_applied: List[str]
    data_lineage: Dict[str, Any]

    # Human-in-loop
    requires_human_approval: bool
    human_feedback: List[Dict[str, Any]]
    auto_approval_enabled: bool


# =============================================================================
# GUARDRAILS SYSTEM
# =============================================================================


class ContentGuardrails:
    """Content safety and compliance guardrails."""

    def __init__(self):
        self.blocked_patterns = [
            r"(confidential|proprietary|trade secret)",
            r"(insider information|material non-public)",
            r"(illegal|fraudulent|deceptive)",
            r"(discriminatory|biased hiring)",
        ]
        self.sensitive_topics = [
            "personal financial data",
            "private company financials",
            "unreleased product details",
            "employee personal info",
        ]

    def check_input_safety(self, content: str, context: Dict[str, Any]) -> SafetyCheck:
        """Check input content for safety violations."""
        violations = []
        risk_level = RiskLevel.LOW

        # Check for inappropriate content
        content_lower = content.lower()

        # Privacy violations
        if any(
            pattern in content_lower
            for pattern in ["ssn", "social security", "bank account"]
        ):
            violations.append(GuardrailViolation.DATA_PRIVACY)
            risk_level = RiskLevel.HIGH

        # Inappropriate requests
        if any(
            word in content_lower
            for word in ["hack", "exploit", "manipulate investors"]
        ):
            violations.append(GuardrailViolation.INAPPROPRIATE_CONTENT)
            risk_level = RiskLevel.CRITICAL

        # Content policy violations
        if len(content) > 5000:  # Prevent prompt injection
            violations.append(GuardrailViolation.CONTENT_POLICY)
            risk_level = max(risk_level, RiskLevel.MEDIUM)

        return SafetyCheck(
            passed=len(violations) == 0,
            risk_level=risk_level,
            violations=violations,
            explanation=f"Input safety check: {len(violations)} violations found",
        )

    def check_output_safety(self, content: str, context: Dict[str, Any]) -> SafetyCheck:
        """Check output content for safety and quality."""
        violations = []
        risk_level = RiskLevel.LOW

        # Check for potential hallucinations
        if self._detect_hallucination_risk(content):
            violations.append(GuardrailViolation.HALLUCINATION_RISK)
            risk_level = RiskLevel.MEDIUM

        # Check for bias
        if self._detect_bias(content):
            violations.append(GuardrailViolation.BIAS_DETECTED)
            risk_level = max(risk_level, RiskLevel.MEDIUM)

        # Check for sensitive information leakage
        if any(topic in content.lower() for topic in self.sensitive_topics):
            violations.append(GuardrailViolation.DATA_PRIVACY)
            risk_level = RiskLevel.HIGH

        return SafetyCheck(
            passed=risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM],
            risk_level=risk_level,
            violations=violations,
            explanation=f"Output safety check: {len(violations)} violations found",
        )

    def _detect_hallucination_risk(self, content: str) -> bool:
        """Detect potential hallucinations in generated content."""
        # Simple heuristics - in production, use more sophisticated methods
        hallucination_indicators = [
            "definitely will succeed",
            "guaranteed returns",
            "100% certain",
            "never fails",
            "always works",
        ]
        return any(
            indicator in content.lower() for indicator in hallucination_indicators
        )

    def _detect_bias(self, content: str) -> bool:
        """Detect potential bias in generated content."""
        bias_indicators = [
            "obviously better",
            "clearly inferior",
            "all successful founders are",
            "typical female entrepreneur",
        ]
        return any(indicator in content.lower() for indicator in bias_indicators)


class RateLimiter:
    """Rate limiting for API calls and actions."""

    def __init__(self):
        self.limits = {
            ActionType.SEARCH: {"count": 0, "limit": 10, "window": 3600},  # 10/hour
            ActionType.GENERATE: {"count": 0, "limit": 50, "window": 3600},  # 50/hour
            ActionType.EXTERNAL_CALL: {
                "count": 0,
                "limit": 5,
                "window": 3600,
            },  # 5/hour
        }
        self.reset_times = {}

    def check_rate_limit(self, action_type: ActionType) -> SafetyCheck:
        """Check if action is within rate limits."""
        now = datetime.now()

        # Reset counters if window expired
        if action_type not in self.reset_times:
            self.reset_times[action_type] = now + timedelta(
                seconds=self.limits[action_type]["window"]
            )
        elif now > self.reset_times[action_type]:
            self.limits[action_type]["count"] = 0
            self.reset_times[action_type] = now + timedelta(
                seconds=self.limits[action_type]["window"]
            )

        # Check limit
        current_count = self.limits[action_type]["count"]
        limit = self.limits[action_type]["limit"]

        if current_count >= limit:
            return SafetyCheck(
                passed=False,
                risk_level=RiskLevel.HIGH,
                violations=[GuardrailViolation.RATE_LIMIT],
                explanation=f"Rate limit exceeded for {action_type.value}: {current_count}/{limit}",
            )

        # Increment counter
        self.limits[action_type]["count"] += 1

        return SafetyCheck(
            passed=True,
            risk_level=RiskLevel.LOW,
            explanation=f"Rate limit OK: {current_count + 1}/{limit}",
        )


# =============================================================================
# HUMAN-IN-THE-LOOP SYSTEM
# =============================================================================


class HumanInTheLoopManager:
    """Manages human approval workflows."""

    def __init__(self):
        self.pending_approvals = {}
        self.approval_history = []
        self.auto_approval_rules = {
            RiskLevel.LOW: True,
            RiskLevel.MEDIUM: False,  # Require human approval
            RiskLevel.HIGH: False,
            RiskLevel.CRITICAL: False,
        }

    def requires_approval(
        self, action_type: ActionType, safety_check: SafetyCheck
    ) -> bool:
        """Determine if action requires human approval."""
        # Auto-approve low-risk actions
        if safety_check.risk_level == RiskLevel.LOW:
            return False

        # Always require approval for critical risk
        if safety_check.risk_level == RiskLevel.CRITICAL:
            return True

        # Apply custom rules based on action type
        if action_type == ActionType.EXTERNAL_CALL:
            return True  # Always require approval for external calls

        if action_type == ActionType.GENERATE and len(safety_check.violations) > 0:
            return True

        return not self.auto_approval_rules.get(safety_check.risk_level, False)

    def request_approval(
        self,
        action_type: ActionType,
        content: str,
        safety_check: SafetyCheck,
        timeout: int = 300,
    ) -> HumanApproval:
        """Request human approval for an action."""
        request_id = hashlib.md5(
            f"{action_type.value}{content}{datetime.now()}".encode()
        ).hexdigest()[:8]

        approval = HumanApproval(
            request_id=request_id,
            action_type=action_type,
            content=content,
            risk_assessment=safety_check,
            timestamp=datetime.now(),
            timeout_seconds=timeout,
        )

        self.pending_approvals[request_id] = approval
        return approval

    def simulate_human_approval(self, approval: HumanApproval) -> HumanApproval:
        """Simulate human approval for demo purposes."""
        # In production, this would integrate with actual human reviewers
        console.print(f"\n🚨 HUMAN APPROVAL REQUIRED", style="bold red")

        table = Table()
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Request ID", approval.request_id)
        table.add_row("Action Type", approval.action_type.value)
        table.add_row("Risk Level", approval.risk_assessment.risk_level.value)
        table.add_row(
            "Violations",
            ", ".join([v.value for v in approval.risk_assessment.violations]),
        )
        table.add_row(
            "Content Preview",
            (
                approval.content[:100] + "..."
                if len(approval.content) > 100
                else approval.content
            ),
        )

        console.print(table)

        # Simulate approval logic
        if approval.risk_assessment.risk_level == RiskLevel.CRITICAL:
            approval.approved = False
            approval.human_feedback = "DENIED: Critical risk level detected"
        elif len(approval.risk_assessment.violations) > 2:
            approval.approved = False
            approval.human_feedback = "DENIED: Multiple violations detected"
        else:
            approval.approved = True
            approval.human_feedback = "APPROVED: Acceptable risk level"

        console.print(
            f"Decision: {'✅ APPROVED' if approval.approved else '❌ DENIED'}",
            style="green" if approval.approved else "red",
        )

        return approval


# =============================================================================
#  VC PITCH ASSISTANT WITH GUARDRAILS
# =============================================================================


class EnhancedVCPitchAssistantAgent:
    def __init__(self):
        """Initialize enhanced agent with guardrails and HITL."""
        self.llm = llm
        self.console = console

        # Initialize guardrails
        self.content_guardrails = ContentGuardrails()
        self.rate_limiter = RateLimiter()
        self.hitl_manager = HumanInTheLoopManager()

        # Initialize tools
        self._initialize_tools()
        self._setup_enhanced_prompts()
        self.workflow = self._build_enhanced_workflow()

        # Audit trail
        self.audit_trail = []

    def _initialize_tools(self):
        """Initialize search tools with error handling."""
        try:
            self.vector_search_tool = VectorStoreSearchTool()
            console.print("✅ Vector search tool initialized", style="green")
        except Exception as e:
            console.print(f"❌ Vector search tool error: {e}", style="red")
            self.vector_search_tool = None

    def _setup_enhanced_prompts(self):
        """Setup enhanced prompt templates."""
        self.query_classifier_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
You are a VC query classifier. Analyze this query and classify it.

Query: "{query}"

Categories:
- pitch_deck: Building, reviewing, or improving pitch decks
- funding: Fundraising strategies, timing, or processes  
- investor: Targeting investors, investor relations, or networking
- market: Market analysis, competitive landscape, or positioning
- metrics: KPIs, financial projections, or performance tracking
- general: Other startup/VC related questions

IMPORTANT SAFETY CHECKS:
- If the query requests confidential information, classify as "inappropriate"
- If the query asks for guaranteed predictions, classify as "inappropriate"
- If the query contains suspicious patterns, classify as "inappropriate"

Respond with ONLY the category name.
""",
        )

        self.answer_generation_prompt = PromptTemplate(
            input_variables=[
                "query",
                "query_type",
                "pitch_deck_context",
                "funding_news_context",
                "safety_context",
            ],
            template="""
You are an expert VC Pitch Assistant. Generate a helpful, accurate, and safe response.

USER QUERY: {query}
QUERY TYPE: {query_type}

PITCH DECK INSIGHTS: {pitch_deck_context}
FUNDING LANDSCAPE: {funding_news_context}
SAFETY CONTEXT: {safety_context}

RESPONSE GUIDELINES:
- Provide specific, actionable advice
- Base recommendations on evidence from search results
- Avoid guarantees or definitive predictions
- Include appropriate disclaimers for financial advice
- Maintain professional, unbiased tone
- Flag any limitations in the available data

SAFETY REQUIREMENTS:
- Do not provide confidential information
- Avoid discriminatory language or bias
- Include appropriate risk disclaimers
- Do not guarantee specific outcomes

Format your response with:
1. **Executive Summary** (2-3 sentences)
2. **Key Recommendations** (3-5 actionable items)
3. **Supporting Evidence** (from search results)
4. **Risk Considerations** (potential challenges)
5. **Next Steps** (specific actions)
6. **Disclaimers** (limitations and risks)

RESPONSE:
""",
        )

    def input_safety_node(self, state: AgentState) -> Dict[str, Any]:
        """Check input safety and apply guardrails."""
        try:
            console.print("🛡️ Checking input safety...", style="blue")

            # Perform safety checks
            safety_check = self.content_guardrails.check_input_safety(
                state["query"], {"user_context": state.get("conversation_history", [])}
            )

            # Check rate limits
            rate_check = self.rate_limiter.check_rate_limit(ActionType.GENERATE)

            # Combine safety checks
            all_violations = safety_check.violations + rate_check.violations
            max_risk = max(
                safety_check.risk_level, rate_check.risk_level, key=lambda x: x.value
            )

            combined_check = SafetyCheck(
                passed=safety_check.passed and rate_check.passed,
                risk_level=max_risk,
                violations=all_violations,
                explanation=f"Input safety: {safety_check.explanation}. Rate limit: {rate_check.explanation}",
            )

            # Log audit trail
            self._log_audit_event(
                "input_safety_check",
                {
                    "query": state["query"][:100],
                    "safety_check": {
                        "passed": combined_check.passed,
                        "risk_level": combined_check.risk_level.value,
                        "violations": [v.value for v in combined_check.violations],
                    },
                },
            )

            if not combined_check.passed:
                console.print(
                    f"❌ Input safety check failed: {combined_check.explanation}",
                    style="red",
                )
                return {
                    "safety_checks": [combined_check],
                    "risk_level": combined_check.risk_level.value,
                    "guardrail_violations": [
                        v.value for v in combined_check.violations
                    ],
                    "error_message": f"Request blocked due to safety concerns: {combined_check.explanation}",
                }

            console.print("✅ Input safety check passed", style="green")
            return {
                "safety_checks": [combined_check],
                "risk_level": combined_check.risk_level.value,
                "guardrail_violations": [],
                "requires_human_approval": False,
            }

        except Exception as e:
            error_msg = f"Input safety check failed: {str(e)}"
            console.print(f"❌ {error_msg}", style="red")
            return {"error_message": error_msg}

    def query_classification_node(self, state: AgentState) -> Dict[str, Any]:
        """Enhanced query classification with safety checks."""
        try:
            # Skip if previous safety check failed
            if state.get("error_message"):
                return {"query_type": "blocked"}

            console.print("🔍 Classifying query type...", style="blue")

            chain = self.query_classifier_prompt | self.llm
            query_type = chain.invoke({"query": state["query"]}).strip().lower()

            # Block inappropriate queries
            if query_type == "inappropriate":
                return {
                    "query_type": "blocked",
                    "error_message": "Query blocked due to inappropriate content",
                }

            # Enhance query based on type
            enhanced_query = self._enhance_query(state["query"], query_type)

            console.print(f"✅ Query classified as: {query_type}", style="green")

            return {
                "query_type": query_type,
                "query": enhanced_query,
                "original_query": state["query"],
            }

        except Exception as e:
            console.print(f"❌ Query classification failed: {e}", style="red")
            return {
                "query_type": "error",
                "error_message": f"Classification failed: {str(e)}",
            }

    def _enhance_query(self, original_query: str, query_type: str) -> str:
        """Enhance query based on classification."""
        enhancements = {
            "pitch_deck": f"{original_query} pitch deck slides template structure best practices",
            "funding": f"{original_query} startup fundraising venture capital series A seed round",
            "investor": f"{original_query} VC investors funding startup investment strategy",
            "market": f"{original_query} market analysis competitive landscape startup positioning",
            "metrics": f"{original_query} startup KPI metrics financial projections benchmarks",
        }
        return enhancements.get(query_type, original_query)

    def parallel_search_node(self, state: AgentState) -> Dict[str, Any]:
        """Enhanced parallel search with guardrails."""
        try:
            # Skip if blocked
            if state.get("query_type") == "blocked" or state.get("error_message"):
                return {"search_results": {"vector": [], "web": []}}

            console.print(
                "🔍 Executing parallel search with safety checks...", style="blue"
            )

            # Check rate limits for search
            search_rate_check = self.rate_limiter.check_rate_limit(ActionType.SEARCH)
            if not search_rate_check.passed:
                return {
                    "search_results": {"vector": [], "web": []},
                    "error_message": "Search rate limit exceeded",
                }

            search_results = {"vector": [], "web": []}
            search_metadata = {"vector_count": 0, "web_count": 0, "search_time": 0}

            import time

            start_time = time.time()

            # Parallel search with error handling
            with ThreadPoolExecutor(max_workers=2) as executor:
                vector_future = executor.submit(
                    self._safe_vector_search, state["query"], state["query_type"]
                )
                web_future = executor.submit(
                    self._safe_web_search, state["query"], state["query_type"]
                )

                vector_results = vector_future.result()
                web_results = web_future.result()

                search_results["vector"] = vector_results
                search_results["web"] = web_results

            search_metadata.update(
                {
                    "vector_count": len(vector_results),
                    "web_count": len(web_results),
                    "search_time": time.time() - start_time,
                }
            )

            # Log search audit
            self._log_audit_event(
                "search_completed",
                {
                    "query_type": state["query_type"],
                    "results_count": search_metadata["vector_count"]
                    + search_metadata["web_count"],
                    "search_time": search_metadata["search_time"],
                },
            )

            console.print(
                f"✅ Safe search completed: {search_metadata['vector_count']} pitch decks, "
                f"{search_metadata['web_count']} articles in {search_metadata['search_time']:.2f}s",
                style="green",
            )

            return {
                "search_results": search_results,
                "search_metadata": search_metadata,
            }

        except Exception as e:
            error_msg = f"Enhanced search failed: {str(e)}"
            console.print(f"❌ {error_msg}", style="red")
            return {
                "search_results": {"vector": [], "web": []},
                "error_message": error_msg,
            }

    def _safe_vector_search(self, query: str, query_type: str) -> List[Dict[str, Any]]:
        """Vector search with safety checks."""
        if not self.vector_search_tool:
            return []

        try:
            # Apply content filtering to search query
            filtered_query = self._filter_search_query(query)

            k_values = {
                "pitch_deck": 6,
                "funding": 4,
                "investor": 3,
                "market": 5,
                "metrics": 4,
                "general": 5,
            }

            k = k_values.get(query_type, 5)
            documents = self.vector_search_tool.search(filtered_query, k=k)

            # Filter results for safety
            safe_results = self._filter_search_results(
                self.vector_search_tool.format_results(
                    documents, max_content_length=400
                )
            )

            return safe_results

        except Exception as e:
            logger.warning(f"Safe vector search error: {e}")
            return []

    def _safe_web_search(self, query: str, query_type: str) -> List[Dict[str, Any]]:
        """Web search with safety checks."""
        try:
            # Apply content filtering
            filtered_query = self._filter_search_query(query)

            web_queries = {
                "pitch_deck": f"{filtered_query} pitch deck template 2024 2025 best practices",
                "funding": f"{filtered_query} startup funding rounds 2024 2025 venture capital trends",
                "investor": f"{filtered_query} VC investors startup funding 2024 2025 strategy",
                "market": f"{filtered_query} market trends startup industry 2024 2025 analysis",
                "metrics": f"{filtered_query} startup metrics benchmarks KPI 2024 2025",
            }

            enhanced_query = web_queries.get(
                query_type, f"{filtered_query} startup 2024 2025"
            )

            search_results = search_tavily(
                query=enhanced_query, max_res=6, search_depth="advanced"
            )

            if search_results.get("error"):
                return []

            # Filter results for safety
            safe_results = self._filter_search_results(
                search_results.get("results", [])
            )
            return safe_results

        except Exception as e:
            logger.warning(f"Safe web search error: {e}")
            return []

    def _filter_search_query(self, query: str) -> str:
        """Filter search query for safety."""
        # Remove potentially problematic terms
        blocked_terms = ["confidential", "insider", "private", "leak", "hack"]
        filtered_query = query

        for term in blocked_terms:
            filtered_query = filtered_query.replace(term, "")

        return filtered_query.strip()

    def _filter_search_results(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter search results for safety and quality."""
        safe_results = []

        for result in results:
            content = result.get("content", "").lower()

            # Skip results with sensitive content
            if any(
                term in content for term in ["confidential", "proprietary", "insider"]
            ):
                continue

            # Skip low-quality results
            if len(content.strip()) < 50:
                continue

            safe_results.append(result)

        return safe_results

    def generate_answer_node(self, state: AgentState) -> Dict[str, Any]:
        """Enhanced answer generation with safety checks and HITL."""
        try:
            # Skip if blocked
            if state.get("query_type") == "blocked" or state.get("error_message"):
                return {
                    "final_answer": "I cannot process this request due to safety concerns.",
                    "requires_human_approval": False,
                }

            console.print("🤖 Generating response with safety checks...", style="blue")

            # Prepare context
            pitch_deck_context = self._format_pitch_deck_context(
                state["search_results"]["vector"]
            )
            funding_news_context = self._format_funding_news_context(
                state["search_results"]["web"]
            )
            safety_context = self._format_safety_context(state.get("safety_checks", []))

            # Generate initial response
            chain = self.answer_generation_prompt | self.llm
            response = chain.invoke(
                {
                    "query": state["original_query"],
                    "query_type": state["query_type"],
                    "pitch_deck_context": pitch_deck_context,
                    "funding_news_context": funding_news_context,
                    "safety_context": safety_context,
                }
            )

            # Check output safety
            output_safety = self.content_guardrails.check_output_safety(
                response,
                {
                    "query_type": state["query_type"],
                    "search_results": state["search_results"],
                },
            )

            # Determine if human approval needed
            requires_approval = self.hitl_manager.requires_approval(
                ActionType.GENERATE, output_safety
            )

            if requires_approval:
                console.print("🚨 Response requires human approval", style="yellow")

                approval = self.hitl_manager.request_approval(
                    ActionType.GENERATE, response, output_safety
                )

                # Simulate approval process
                approved_request = self.hitl_manager.simulate_human_approval(approval)

                if not approved_request.approved:
                    return {
                        "final_answer": f"Response blocked: {approved_request.human_feedback}",
                        "requires_human_approval": True,
                        "safety_checks": state.get("safety_checks", [])
                        + [output_safety],
                    }

                # Add human feedback to response
                response = f"{response}\n\n*Human Reviewer Note: {approved_request.human_feedback}*"

            # Log generation audit
            self._log_audit_event(
                "response_generated",
                {
                    "query_type": state["query_type"],
                    "output_safety": {
                        "passed": output_safety.passed,
                        "risk_level": output_safety.risk_level.value,
                        "violations": [v.value for v in output_safety.violations],
                    },
                    "required_approval": requires_approval,
                    "response_length": len(response),
                },
            )

            console.print("✅ Safe response generated", style="green")

            return {
                "final_answer": response,
                "requires_human_approval": requires_approval,
                "safety_checks": state.get("safety_checks", []) + [output_safety],
            }

        except Exception as e:
            error_msg = f"Enhanced answer generation failed: {str(e)}"
            console.print(f"❌ {error_msg}", style="red")
            return {
                "final_answer": f"I apologize, but I encountered an error: {error_msg}",
                "error_message": error_msg,
            }

    def _format_pitch_deck_context(self, vector_results: List[Dict[str, Any]]) -> str:
        """Format pitch deck database results with safety filtering."""
        if not vector_results:
            return "No relevant pitch deck examples found in database."

        context_parts = []
        for i, result in enumerate(vector_results[:4], 1):
            content = result.get("content", "")
            metadata = result.get("metadata", {})

            # Safety filter content
            if any(term in content.lower() for term in ["confidential", "proprietary"]):
                continue

            startup_name = metadata.get("startup_name", "Anonymous Startup")
            stage = metadata.get("funding_stage", "Unknown Stage")
            industry = metadata.get("industry", "Unknown Industry")

            context_parts.append(
                f"EXAMPLE {i}: {startup_name} ({stage} - {industry})\n"
                f"Key Insights: {content[:300]}...\n"
            )

        return (
            "\n".join(context_parts) if context_parts else "No safe examples available."
        )

    def _format_funding_news_context(self, web_results: List[Dict[str, Any]]) -> str:
        """Format current funding news with safety filtering."""
        if not web_results:
            return "No current funding news available."

        context_parts = []
        for i, result in enumerate(web_results[:4], 1):
            title = result.get("title", "Unknown Title")
            content = result.get("content", "")

            # Safety filter content
            if any(
                term in content.lower()
                for term in ["confidential", "insider", "proprietary"]
            ):
                continue

            context_parts.append(
                f"NEWS {i}: {title}\n" f"Summary: {content[:250]}...\n"
            )

        return (
            "\n".join(context_parts)
            if context_parts
            else "No safe news content available."
        )

    def _format_safety_context(self, safety_checks: List[SafetyCheck]) -> str:
        """Format safety context for prompt."""
        if not safety_checks:
            return "No safety concerns identified."

        context_parts = []
        for check in safety_checks:
            if check.violations:
                context_parts.append(f"Safety Alert: {check.explanation}")

        return (
            "\n".join(context_parts) if context_parts else "All safety checks passed."
        )

    def _log_audit_event(self, event_type: str, details: Dict[str, Any]):
        """Log audit events for compliance."""
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details,
            "session_id": getattr(self, "session_id", "unknown"),
        }
        self.audit_trail.append(audit_entry)
        logger.info(f"Audit: {event_type} - {details}")

    def _build_enhanced_workflow(self) -> StateGraph:
        """Build enhanced workflow with guardrails and HITL."""
        workflow = StateGraph(AgentState)

        # Add enhanced nodes
        workflow.add_node("input_safety", self.input_safety_node)
        workflow.add_node("classify_query", self.query_classification_node)
        workflow.add_node("parallel_search", self.parallel_search_node)
        workflow.add_node("generate_answer", self.generate_answer_node)
        workflow.add_node("output_safety", self.output_safety_node)

        # Create enhanced flow with safety checkpoints
        workflow.add_edge(START, "input_safety")
        workflow.add_edge("input_safety", "classify_query")
        workflow.add_edge("classify_query", "parallel_search")
        workflow.add_edge("parallel_search", "generate_answer")
        workflow.add_edge("generate_answer", "output_safety")
        workflow.add_edge("output_safety", END)

        # Compile with memory
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def output_safety_node(self, state: AgentState) -> Dict[str, Any]:
        """Final output safety check and content filtering."""
        try:
            console.print("🛡️ Final output safety check...", style="blue")

            if state.get("error_message"):
                return {"final_answer": state.get("final_answer", "Error occurred")}

            final_answer = state.get("final_answer", "")

            # Apply final content filters
            filtered_answer = self._apply_content_filters(final_answer)

            # Add safety disclaimers
            enhanced_answer = self._add_safety_disclaimers(
                filtered_answer, state["query_type"]
            )

            # Log final audit
            self._log_audit_event(
                "response_delivered",
                {
                    "query_type": state["query_type"],
                    "original_length": len(final_answer),
                    "filtered_length": len(enhanced_answer),
                    "safety_checks_count": len(state.get("safety_checks", [])),
                },
            )

            console.print("✅ Output safety check completed", style="green")

            return {
                "final_answer": enhanced_answer,
                "content_filters_applied": [
                    "profanity_filter",
                    "privacy_filter",
                    "disclaimer_added",
                ],
                "audit_trail": self.audit_trail[-5:],  # Last 5 audit entries
            }

        except Exception as e:
            error_msg = f"Output safety check failed: {str(e)}"
            console.print(f"❌ {error_msg}", style="red")
            return {
                "final_answer": "I apologize, but I cannot provide a safe response at this time.",
                "error_message": error_msg,
            }

    def _apply_content_filters(self, content: str) -> str:
        """Apply final content filters."""
        # Remove any remaining sensitive patterns
        filtered_content = content

        # Pattern replacements for safety
        replacements = {
            r"\b(guaranteed|100% certain|definitely will)\b": "likely to",
            r"\b(never fails|always works)\b": "typically succeeds",
            r"\b(secret|confidential) (information|data)\b": "proprietary insights",
        }

        import re

        for pattern, replacement in replacements.items():
            filtered_content = re.sub(
                pattern, replacement, filtered_content, flags=re.IGNORECASE
            )

        return filtered_content

    def _add_safety_disclaimers(self, content: str, query_type: str) -> str:
        """Add appropriate safety disclaimers."""
        disclaimers = {
            "funding": "\n\n**Disclaimer**: This information is for educational purposes only and should not be considered as financial or investment advice. Please consult with qualified professionals before making funding decisions.",
            "investor": "\n\n**Disclaimer**: Investor information is based on publicly available data. Always conduct your own due diligence and verify investor requirements before reaching out.",
            "metrics": "\n\n**Disclaimer**: Metrics and benchmarks may vary significantly by industry, stage, and market conditions. Use these as general guidelines only.",
            "pitch_deck": "\n\n**Disclaimer**: Pitch deck recommendations are based on general best practices. Customize your approach based on your specific industry and audience.",
            "market": "\n\n**Disclaimer**: Market analysis is based on available data and trends. Market conditions can change rapidly, so verify current information independently.",
        }

        disclaimer = disclaimers.get(
            query_type,
            "\n\n**Disclaimer**: This information is provided for educational purposes. Please verify all information independently.",
        )

        return content + disclaimer

    def run_enhanced(
        self,
        query: str,
        session_id: str = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Execute enhanced workflow with full guardrails and HITL."""
        try:
            self.session_id = (
                session_id or f"session_{hashlib.md5(query.encode()).hexdigest()[:8]}"
            )

            console.print(f"🚀 Enhanced Processing: '{query}'", style="bold blue")
            console.print(f"Session ID: {self.session_id}", style="dim")

            # Initialize enhanced state
            initial_state = {
                "query": query,
                "original_query": query,
                "query_type": "",
                "search_results": {"vector": [], "web": []},
                "final_answer": "",
                "conversation_history": conversation_history or [],
                "error_message": None,
                "search_metadata": {},
                "safety_checks": [],
                "pending_approvals": [],
                "risk_level": "low",
                "guardrail_violations": [],
                "audit_trail": [],
                "content_filters_applied": [],
                "data_lineage": {},
                "requires_human_approval": False,
                "human_feedback": [],
                "auto_approval_enabled": True,
            }

            # Execute enhanced workflow
            config = {"configurable": {"thread_id": self.session_id}}
            final_state = self.workflow.invoke(initial_state, config)

            # Update conversation history
            if conversation_history is not None:
                conversation_history.append(
                    {
                        "query": query,
                        "answer": final_state.get("final_answer", "")[:500],
                        "session_id": self.session_id,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            # Prepare enhanced response
            search_meta = final_state.get("search_metadata", {})
            safety_summary = self._summarize_safety_checks(
                final_state.get("safety_checks", [])
            )

            console.print("🎉 Enhanced response ready!", style="bold green")

            return {
                "answer": final_state.get(
                    "final_answer", "Unable to generate safe response."
                ),
                "query_type": final_state.get("query_type", "unknown"),
                "pitch_deck_count": search_meta.get("vector_count", 0),
                "news_count": search_meta.get("web_count", 0),
                "processing_time": search_meta.get("search_time", 0),
                "error": final_state.get("error_message"),
                # Enhanced safety and compliance info
                "safety_summary": safety_summary,
                "risk_level": final_state.get("risk_level", "unknown"),
                "guardrail_violations": final_state.get("guardrail_violations", []),
                "required_human_approval": final_state.get(
                    "requires_human_approval", False
                ),
                "content_filters_applied": final_state.get(
                    "content_filters_applied", []
                ),
                "audit_trail_summary": len(self.audit_trail),
                "session_id": self.session_id,
                # Compliance and transparency
                "data_sources": {
                    "pitch_deck_database": search_meta.get("vector_count", 0) > 0,
                    "web_search": search_meta.get("web_count", 0) > 0,
                    "llm_generation": True,
                },
                "processing_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "model_version": "gemini-1.5-flash",
                    "safety_checks_passed": len(
                        [c for c in final_state.get("safety_checks", []) if c.passed]
                    ),
                    "total_safety_checks": len(final_state.get("safety_checks", [])),
                },
            }

        except Exception as e:
            error_msg = f"Enhanced workflow failed: {str(e)}"
            console.print(f"💥 {error_msg}", style="bold red")

            # Log critical error
            self._log_audit_event(
                "critical_error",
                {
                    "error": error_msg,
                    "query": query[:100],
                    "session_id": self.session_id,
                },
            )

            return {
                "answer": f"I apologize, but I encountered a critical error: {error_msg}",
                "query_type": "error",
                "pitch_deck_count": 0,
                "news_count": 0,
                "processing_time": 0,
                "error": error_msg,
                "safety_summary": "Error occurred during processing",
                "risk_level": "critical",
                "session_id": self.session_id,
            }

    def _summarize_safety_checks(self, safety_checks: List[SafetyCheck]) -> str:
        """Summarize safety check results."""
        if not safety_checks:
            return "No safety checks performed"

        passed = len([c for c in safety_checks if c.passed])
        total = len(safety_checks)
        violations = sum(len(c.violations) for c in safety_checks)

        risk_levels = [c.risk_level.value for c in safety_checks]
        max_risk = max(risk_levels) if risk_levels else "low"

        return f"Safety: {passed}/{total} checks passed, {violations} violations, max risk: {max_risk}"

    def get_audit_report(self, session_id: str = None) -> Dict[str, Any]:
        """Generate audit report for compliance."""
        filtered_audit = self.audit_trail
        if session_id:
            filtered_audit = [
                e for e in self.audit_trail if e.get("session_id") == session_id
            ]

        report = {
            "report_generated": datetime.now().isoformat(),
            "session_id": session_id,
            "total_events": len(filtered_audit),
            "event_summary": {},
            "safety_events": [],
            "errors": [],
            "compliance_status": "compliant",
        }

        # Summarize events
        for event in filtered_audit:
            event_type = event["event_type"]
            report["event_summary"][event_type] = (
                report["event_summary"].get(event_type, 0) + 1
            )

            # Track safety-related events
            if "safety" in event_type or "error" in event_type:
                report["safety_events"].append(event)

            if event_type == "critical_error":
                report["errors"].append(event)
                report["compliance_status"] = "needs_review"

        return report


# =============================================================================
# DEMO AND TESTING
# =============================================================================


def demo_enhanced_agent():
    """Demonstrate the enhanced VC Pitch Assistant with guardrails."""
    try:
        console.print("🎯 Initializing Enhanced VC Pitch Assistant", style="bold blue")
        agent = EnhancedVCPitchAssistantAgent()

        # Test queries with different risk levels
        test_scenarios = [
          
            {
                "query": "tell me about the vapi pitch deck",
                "expected_risk": "low",
                "description": "",
            }
        ]

        conversation_history = []

        for i, scenario in enumerate(test_scenarios, 1):
            console.print(f"\n{'='*80}", style="bold")
            console.print(
                f"TEST SCENARIO {i}: {scenario['description']}", style="bold cyan"
            )
            console.print(
                f"Expected Risk Level: {scenario['expected_risk']}", style="yellow"
            )

            result = agent.run_enhanced(
                scenario["query"],
                session_id=f"test_session_{i}",
                conversation_history=conversation_history,
            )

            # Display results
            console.print(f"\nQuery: {scenario['query']}", style="bold white")

            # Create results table
            table = Table()
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")

            table.add_row("Query Type", result["query_type"])
            table.add_row("Risk Level", result["risk_level"])
            table.add_row("Safety Summary", result["safety_summary"])
            table.add_row(
                "Human Approval Required", str(result["required_human_approval"])
            )
            table.add_row("Processing Time", f"{result['processing_time']:.2f}s")
            table.add_row(
                "Sources",
                f"{result['pitch_deck_count']} pitch decks, {result['news_count']} news",
            )
            table.add_row(
                "Filters Applied", ", ".join(result["content_filters_applied"])
            )

            console.print(table)

            # Show response
            if result["error"]:
                console.print(f"\n❌ Error: {result['error']}", style="bold red")
            else:
                console.print(f"\n📊 Response:", style="bold green")
                console.print(
                    Panel(
                        result["answer"]
                        
                    )
                )

            # Show guardrail violations if any
            if result["guardrail_violations"]:
                console.print(
                    f"\n⚠️ Guardrail Violations: {', '.join(result['guardrail_violations'])}",
                    style="bold yellow",
                )

            # Brief pause between scenarios
            import time

            time.sleep(1)

        # Generate final audit report
        console.print(f"\n{'='*80}", style="bold")
        console.print("📋 COMPLIANCE AUDIT REPORT", style="bold blue")

        audit_report = agent.get_audit_report()

        audit_table = Table()
        audit_table.add_column("Audit Metric", style="cyan")
        audit_table.add_column("Value", style="white")

        audit_table.add_row("Total Events", str(audit_report["total_events"]))
        audit_table.add_row("Safety Events", str(len(audit_report["safety_events"])))
        audit_table.add_row("Errors", str(len(audit_report["errors"])))
        audit_table.add_row("Compliance Status", audit_report["compliance_status"])
        audit_table.add_row("Report Generated", audit_report["report_generated"])

        console.print(audit_table)

        # Show event summary
        if audit_report["event_summary"]:
            console.print("\n📊 Event Summary:", style="bold yellow")
            for event_type, count in audit_report["event_summary"].items():
                console.print(f"  • {event_type}: {count}")

        console.print("\n🎉 Enhanced demo completed successfully!", style="bold green")

    except Exception as e:
        console.print(f"💥 Demo failed: {e}", style="bold red")


def main():
    """Main function to run the enhanced demo."""
    demo_enhanced_agent()


if __name__ == "__main__":
    main()
