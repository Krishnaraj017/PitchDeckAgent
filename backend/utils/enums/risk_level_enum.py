from enum import Enum


class RiskLevel(Enum):
    """Risk levels for content and actions."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"