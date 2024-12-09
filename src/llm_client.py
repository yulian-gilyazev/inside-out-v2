from typing import List, Dict

import time
from openai import OpenAI, OpenAIError
from loguru import logger

from src.schema.llm_config import LLMConfig


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self.config = config

    def chat(self, messages, max_tokens=None, retries=3, backoff_factor=0.3):
        if not max_tokens:
            max_tokens = self.config.max_tokens

        for i in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                )
                return response.choices[0]

            except Exception as e:
                logger.error(f"Request failed with error {e}. Trying again.")
                wait_time = backoff_factor * (2 ** i)
                time.sleep(wait_time)

                if i == retries - 1:
                    logger.error("All retries failed.")
                    raise

        raise ConnectionError(f"Failed to connect to {self.config.base_url}.")


class DialogueManager:
    def __init__(self, client1: LLMClient, client2: LLMClient, first_system_prompt: str, second_system_prompt: str):
        self.client1 = client1
        self.client2 = client2
        self.first_system_prompt = first_system_prompt
        self.second_system_prompt = second_system_prompt

    def communicate(self, n_rounds: int) -> List[Dict[str, str]]:
        history1 = [{"role": "system", "content": self.first_system_prompt}]
        history2 = [{"role": "system", "content": self.second_system_prompt}]
        for _ in range(n_rounds):
            message = self.client1.chat(history1).message
            history1.append({"role": "assistant", "content": message.content})
            history2.append({"role": "user", "content": message.content})
            message = self.client1.chat(history2).message
            history2.append({"role": "assistant", "content": message.content})
            history1.append({"role": "user", "content": message.content})
        dialogue = [{"role": "first" if item["role"] == "assistant" else "second", "content": item["content"]} for item in history1[1:]]
        return dialogue