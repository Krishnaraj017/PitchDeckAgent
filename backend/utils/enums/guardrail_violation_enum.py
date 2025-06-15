from enum import Enum


class GuardrailViolation(Enum):
    """Types of guardrail violations."""

    INAPPROPRIATE_CONTENT = "inappropriate_content"
    DATA_PRIVACY = "data_privacy"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    RATE_LIMIT = "rate_limit"
    CONTENT_POLICY = "content_policy"
    HALLUCINATION_RISK = "hallucination_risk"
    BIAS_DETECTED = "bias_detected"
