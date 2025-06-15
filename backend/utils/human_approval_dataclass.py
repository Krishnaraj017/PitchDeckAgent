from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from utils.enums.action_type_enum import ActionType
from utils.security_dataclass import SafetyCheck
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

