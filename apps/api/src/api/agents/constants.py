from openai.types import ReasoningEffort

from api.core.types import Provider

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
LLM_TIMEOUT_SECONDS = 45
QDRANT_TIMEOUT_SECONDS = 10
MAX_TOKENS = 2000
MAX_HISTORY_MESSAGES = 10
DEFAULT_TOP_K = 5
DEFAULT_PROVIDER: Provider = Provider.OPENAI
DEFAULT_MODEL_NAME = "gpt-6-sol"
DEFAULT_REASONING_EFFORT: ReasoningEffort = "low"
