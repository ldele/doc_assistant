"""Token usage tracking for LLM calls."""
from langchain_core.callbacks import BaseCallbackHandler


class TokenCounter(BaseCallbackHandler):
    """Tracks tokens across all LLM calls in a session."""

    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    def on_llm_end(self, response, **kwargs):
        self.calls += 1
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg and hasattr(msg, "usage_metadata") and msg.usage_metadata:
                    self.input_tokens += msg.usage_metadata.get("input_tokens", 0)
                    self.output_tokens += msg.usage_metadata.get("output_tokens", 0)

    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def cost_usd(self, input_rate: float = 1.0, output_rate: float = 5.0) -> float:
        """Cost in USD. Rates are per 1M tokens. Defaults to Haiku 4.5 pricing."""
        return (self.input_tokens * input_rate + self.output_tokens * output_rate) / 1_000_000

    def reset(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0