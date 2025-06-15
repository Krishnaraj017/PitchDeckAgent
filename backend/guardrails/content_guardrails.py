
from typing import Any, Dict
from utils.security_dataclass import SafetyCheck
from utils.enums.risk_level_enum import RiskLevel
from utils.enums.guardrail_violation_enum import GuardrailViolation

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

