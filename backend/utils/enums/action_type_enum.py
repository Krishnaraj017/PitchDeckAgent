from enum import Enum
class ActionType(Enum):
    """Types of actions the agent can perform."""

    SEARCH = "search"
    GENERATE = "generate"
    RECOMMEND = "recommend"
    ANALYZE = "analyze"
    EXTERNAL_CALL = "external_call"