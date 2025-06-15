# =============================================================================
# HUMAN-IN-THE-LOOP SYSTEM
# =============================================================================

from datetime import datetime
import hashlib
from rich.table import Table
from rich.console import Console
from utils.enums.risk_level_enum import RiskLevel
from utils.enums.action_type_enum import ActionType
from utils.security_dataclass import SafetyCheck
from utils.human_approval_dataclass import HumanApproval
console = Console()
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

