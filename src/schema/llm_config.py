from typing import Dict, Any
from dataclasses import dataclass
from src.config import *


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model_name: str
    max_tokens: int = None
    temperature: float = None
    top_p: int = None
    input_token_price: float = None
    output_token_price: float = None
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'LlmConfig':
        if "api_key" not in d:
            if d["base_url"] == "https://api.vsegpt.ru/v1":
                d["api_key"] = VSEGPT_API_KEY
            elif d["base_url"] == "https://api.openai.com/v1":
                d["api_key"] = OPENAI_API_KEY
            else:
                raise ValueError("API key is not valid")
        return LLMConfig(**d)

    def get_values_dict(self):
        return {
            "base_url": self.base_url,
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "input_token_price": self.input_token_price,
            "output_token_price": self.output_token_price
        }