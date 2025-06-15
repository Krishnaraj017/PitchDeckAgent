import os
import json
import hashlib
from typing import TypedDict, Any, List, Dict, Optional, Literal
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import logging
import requests  # For LangSmith API calls

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

# =============================================================================
# PRODUCTION CONFIGURATION
# =============================================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "vc-pitch-assistant")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")  # For real human approvals

llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=GOOGLE_API_KEY,
    temperature=0.3,
)

# =============================================================================
# LANGSMITH INTEGRATION (PRODUCTION APPROVAL WORKFLOW)
# =============================================================================


class LangSmithApprovalClient:
    """Production-ready human approval system using LangSmith"""

    def __init__(self):
        self.base_url = "https://api.smith.langchain.com/v1"
        self.headers = {
            "x-api-key": LANGSMITH_API_KEY,
            "Content-Type": "application/json"
        }

    def create_approval_request(
        self,
        action_type: str,
        content: str,
        risk_level: str,
        violations: List[str],
        timeout_minutes: int = 15
    ) -> str:
        """Create a human approval request in LangSmith"""
        payload = {
            "project": LANGSMITH_PROJECT,
            "name": f"Approval Required: {action_type}",
            "input": {
                "action_type": action_type,
                "content_preview": content[:500] + "..." if len(content) > 500 else content,
                "risk_level": risk_level,
                "violations": violations,
                "timeout_minutes": timeout_minutes
            },
            "config": {
                "metadata": {
                    "approval_required": True,
                    "priority": "high" if risk_level in ["high", "critical"] else "medium"
                }
            }
        }

        try:
            response = requests.post(
                f"{self.base_url}/runs",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()["id"]
        except Exception as e:
            logger.error(f"LangSmith approval request failed: {str(e)}")
            raise RuntimeError("Failed to create approval request")

    def check_approval_status(self, run_id: str) -> Dict[str, Any]:
        """Check approval status from LangSmith"""
        try:
            response = requests.get(
                f"{self.base_url}/runs/{run_id}",
                headers=self.headers
            )
            response.raise_for_status()
            data = response.json()

            # LangSmith stores feedback in the 'feedback' field
            if data.get("feedback"):
                return {
                    "approved": data["feedback"].get("value") == "approved",
                    "feedback": data["feedback"].get("comment", ""),
                    "reviewer": data["feedback"].get("created_by", "unknown")
                }
            return {"approved": None, "feedback": ""}
        except Exception as e:
            logger.error(f"LangSmith approval check failed: {str(e)}")
            return {"approved": None, "error": str(e)}

    def notify_slack(self, message: str):
        """Optional: Send Slack notification for urgent approvals"""
        if not SLACK_WEBHOOK_URL:
            return

        payload = {
            "text": f"🚨 VC Assistant Approval Needed\n{message}",
            "username": "VC Approval Bot"
        }
        try:
            requests.post(SLACK_WEBHOOK_URL, json=payload)
        except Exception as e:
            logger.warning(f"Slack notification failed: {str(e)}")

# =============================================================================
# ENHANCED HUMAN APPROVAL SYSTEM (PRODUCTION)
# =============================================================================


class ProductionHumanApprovalManager:
    """Production-grade human approval workflow"""

    def __init__(self):
        self.langsmith = LangSmithApprovalClient()
        self.auto_approval_rules = {
            "low": True,
            "medium": False,
            "high": False,
            "critical": False
        }

    def request_approval(
        self,
        action_type: str,
        content: str,
        safety_check: Dict[str, Any],
        timeout_minutes: int = 15
    ) -> Dict[str, Any]:
        """Create a real human approval request"""
        run_id = self.langsmith.create_approval_request(
            action_type=action_type.value,
            content=content,
            risk_level=safety_check["risk_level"],
            violations=[v.value for v in safety_check["violations"]],
            timeout_minutes=timeout_minutes
        )

        # Send Slack alert for critical items
        if safety_check["risk_level"] in ["high", "critical"]:
            self.langsmith.notify_slack(
                f"Critical approval needed for {action_type.value}\n"
                f"Risk: {safety_check['risk_level']}\n"
                f"Preview: {content[:200]}..."
            )

        return {
            "request_id": run_id,
            "action_type": action_type,
            "content": content,
            "status": "pending",
            "timestamp": datetime.now().isoformat()
        }

    def check_approval(self, request_id: str) -> Dict[str, Any]:
        """Check real approval status from LangSmith"""
        return self.langsmith.check_approval_status(request_id)

# =============================================================================
# UPDATED AGENT WITH PRODUCTION APPROVAL FLOW
# =============================================================================


class ProductionVCPitchAssistant(EnhancedVCPitchAssistantAgent):
    """Production-ready version with LangSmith human approval"""

    def __init__(self):
        super().__init__()
        self.human_approval = ProductionHumanApprovalManager()

    def generate_answer_node(self, state: AgentState) -> Dict[str, Any]:
        """Updated with real approval workflow"""
        try:
            if state.get("query_type") == "blocked" or state.get("error_message"):
                return {
                    "final_answer": "Request blocked by safety system",
                    "requires_human_approval": False
                }

            # ... (previous context preparation code remains the same)

            # Generate initial response
            chain = self.answer_generation_prompt | self.llm
            response = chain.invoke({...})  # Same as original

            # Check output safety
            output_safety = self.content_guardrails.check_output_safety(
                response, {...})
            safety_dict = {
                "passed": output_safety.passed,
                "risk_level": output_safety.risk_level.value,
                "violations": [v.value for v in output_safety.violations],
                "explanation": output_safety.explanation
            }

            # Handle approval workflow
            requires_approval = self.human_approval.requires_approval(
                ActionType.GENERATE,
                safety_dict
            )

            if requires_approval:
                approval_request = self.human_approval.request_approval(
                    ActionType.GENERATE,
                    response,
                    safety_dict
                )

                return {
                    "final_answer": "[PENDING_APPROVAL] Response awaiting human review",
                    "requires_human_approval": True,
                    "approval_request": approval_request,
                    "safety_checks": [...],
                    "pending_response": response  # Store the original response
                }

            # If no approval needed, proceed
            return {
                "final_answer": response,
                "requires_human_approval": False,
                "safety_checks": [...]
            }

        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            return {
                "final_answer": f"Error: {str(e)}",
                "error_message": str(e)
            }

    def approval_check_node(self, state: AgentState) -> Dict[str, Any]:
        """New node to check approval status"""
        if not state.get("requires_human_approval"):
            return state

        approval_request = state.get("approval_request")
        if not approval_request:
            return state

        status = self.human_approval.check_approval(
            approval_request["request_id"])

        if status["approved"] is None:
            # Still waiting
            return {
                "final_answer": "[STILL_PENDING] Waiting for human review...",
                "approval_status": "pending"
            }
        elif status["approved"]:
            # Approved - return the original response
            return {
                "final_answer": state["pending_response"],
                "approval_status": "approved",
                "human_feedback": status["feedback"],
                "requires_human_approval": False
            }
        else:
            # Rejected
            return {
                "final_answer": f"[REJECTED] Human reviewer denied this response: {status['feedback']}",
                "approval_status": "rejected",
                "requires_human_approval": False
            }

    def _build_enhanced_workflow(self) -> StateGraph:
        """Updated workflow with approval check node"""
        workflow = StateGraph(AgentState)

        workflow.add_node("input_safety", self.input_safety_node)
        workflow.add_node("classify_query", self.query_classification_node)
        workflow.add_node("parallel_search", self.parallel_search_node)
        workflow.add_node("generate_answer", self.generate_answer_node)
        workflow.add_node("approval_check",
                          self.approval_check_node)  # New node
        workflow.add_node("output_safety", self.output_safety_node)

        # Updated workflow with approval loop
        workflow.add_edge(START, "input_safety")
        workflow.add_edge("input_safety", "classify_query")
        workflow.add_edge("classify_query", "parallel_search")
        workflow.add_edge("parallel_search", "generate_answer")

        # Conditional edges for approval flow
        workflow.add_conditional_edges(
            "generate_answer",
            lambda state: "approval_check" if state.get(
                "requires_human_approval") else "output_safety",
        )

        workflow.add_edge("approval_check", "output_safety")
        workflow.add_edge("output_safety", END)

        return workflow.compile(checkpointer=MemorySaver())


# =============================================================================
# PRODUCTION DEPLOYMENT EXAMPLE
# =============================================================================
if __name__ == "__main__":
    # Initialize production agent
    agent = ProductionVCPitchAssistant()

    # Example query requiring approval
    query = "How can I guarantee investors 100% returns in my Series A pitch?"

    # Execute workflow
    result = agent.run_enhanced(query)

    if result.get("requires_human_approval"):
        print(
            f"Response pending approval. Request ID: {result['approval_request']['request_id']}")

        # In production, you'd poll for approval status
        import time
        time.sleep(10)  # Simulate waiting

        # Check approval status
        updated_state = agent.workflow.get_state(result["session_id"])
        if updated_state.get("approval_status") == "approved":
            print("Approved response:", updated_state["final_answer"])
        else:
            print("Rejected:", updated_state.get("human_feedback", ""))
    else:
        print("Auto-approved response:", result["answer"])
