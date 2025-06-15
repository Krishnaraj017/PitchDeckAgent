from datetime import datetime, timedelta
from utils.enums.action_type_enum import ActionType
from utils.security_dataclass import SafetyCheck
from utils.enums.risk_level_enum import RiskLevel
from utils.enums.guardrail_violation_enum import GuardrailViolation
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

