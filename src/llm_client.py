from typing import List, Dict

import time
import numpy as np
from openai import OpenAI, ChatCompletion
from loguru import logger
import traceback

from src.schema.llm_config import LLMConfig


class TokenCounterMixin:
    def __init__(self):
        self.input_tokens = []
        self.output_tokens = []

    def update_tokens(self, response: ChatCompletion):
        self.input_tokens.append(response.usage.prompt_tokens)
        self.output_tokens.append(response.usage.completion_tokens)

    def get_total_tokens(self):
        return np.array(self.input_tokens) + np.array(self.output_tokens)

    def get_input_tokens(self):
        return np.array(self.input_tokens)

    def get_output_tokens(self):
        return np.array(self.output_tokens)

    def get_generations_cost(self):
        cost = (self.config.input_token_price * self.get_input_tokens().sum() +
                self.config.output_token_price * self.get_output_tokens().sum())
        return cost


class LLMClient(TokenCounterMixin):
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self.config = config

    def chat(self, messages: List[Dict[str, str]], max_tokens=None, retries=6, backoff_factor=0.3):
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
                self.update_tokens(response)
                return response.choices[0]

            except Exception:
                logger.error(f"Request failed with error.\n{traceback.format_exc()}")
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

    def _prepare_history_for_llm_client(self, dialogue: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if len(dialogue) % 2 == 0:
            history = [{"role": "system", "content": self.first_system_prompt}]
            role = "assistant"
        else:
            history = [{"role": "assistant", "content": self.second_system_prompt}]
            role = "user"
        for i, item in enumerate(dialogue):
            history.append({"role": role, "content": item["content"]})
            role = "assistant" if role == "user" else "user"
        return history

    @staticmethod
    def _llm_client_history_to_dialogue(history: List[Dict[str, str]], interlocutor_idx: int):
        if interlocutor_idx > 1:
            raise ValueError("Interlocutor idx should be 0 or 1.")
        if interlocutor_idx == 0:
            dialogue = [{"role": "first" if item["role"] == "assistant" else "second", "content": item["content"]}
                        for item in history[1:]]
        else:
            dialogue = [{"role": "first" if item["role"] == "user" else "second", "content": item["content"]}
                        for item in history[1:]]
        return dialogue

    def communicate(self, n_rounds: int) -> List[Dict[str, str]]:
        history1 = [{"role": "system", "content": self.first_system_prompt}]
        history2 = [{"role": "system", "content": self.second_system_prompt}]
        for _ in range(n_rounds):
            message = self.client1.chat(history1).message
            history1.append({"role": "assistant", "content": message.content})
            history2.append({"role": "user", "content": message.content})
            message = self.client2.chat(history2).message
            history2.append({"role": "assistant", "content": message.content})
            history1.append({"role": "user", "content": message.content})
        dialogue = self._llm_client_history_to_dialogue(history1, 0)
        return dialogue

    def append_utterance(self, dialogue: List[Dict[str, str]]) -> str:
        history = self._prepare_history_for_llm_client(dialogue)
        if len(dialogue) % 2 == 0:
            client = self.client1
        else:
            client = self.client2
        message = client.chat(history).message
        return message.content
