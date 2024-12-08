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
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'LlmConfig':
        if "api_key" not in d:
            if d["base_url"] == "https://api.vsegpt.ru/v1":
                d["api_key"] = VSEGPT_API_KEY
            else:
                raise ValueError("API key is not valid")
        return LLMConfig(**d)