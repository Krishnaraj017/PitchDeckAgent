from dataclasses import dataclass, field
from typing import List
from utils.enums.risk_level_enum import RiskLevel
from utils.enums.guardrail_violation_enum import GuardrailViolation
@dataclass
class SafetyCheck:
    """Safety check result."""

    passed: bool
    risk_level: RiskLevel
    violations: List[GuardrailViolation] = field(default_factory=list)
    confidence: float = 1.0
    explanation: str = ""
    auto_approved: bool = False


