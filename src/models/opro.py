import re
from loguru import logger
from src.llm_clients import LLMClient
from typing import Callable, Optional, Tuple, List, Union, Any


class OPRO:
    """
    Optimization By Prompting (OPRO).
    https://arxiv.org/abs/2309.03409
    """

    def __init__(
            self, 
            llm_client: LLMClient,
            metric_name: str,
            metaprompt: str, 
            task_prompt: str, 
            check_fn: Optional[Callable[[str], bool]] = None, 
            prompt_tokens: Tuple[str, str] = ("<TEXT>", "</TEXT>")
    ):
        """
        Initialize OPRO.
        
        Args:
            llm_client: Client for interacting with language model
            metric_name: Name of the metric to optimize
            metaprompt: System prompt for LLM
            task_prompt: Task description
            check_fn: Function to check validity of generated prompt
            prompt_tokens: Tokens marking the beginning and end of the prompt
        """
        self.llm_client = llm_client
        self.prompts_history: List[str] = []
        self.reward_history: List[Union[float, int]] = []
        self.metric_name = metric_name
        self.metaprompt = metaprompt
        self.task_prompt = task_prompt
        self.check_fn = check_fn
        self.prompt_tokens = prompt_tokens
        self.re_pattern = rf"{prompt_tokens[0]}(.*?){prompt_tokens[1]}"
        # State: 0 - waiting for step, 1 - waiting for reward
        self._state = 0

    def initialize(self, prompt: str, reward: Union[float, int]) -> None:
        """
        Initialize the history of prompts and rewards.
        
        Args:
            prompt: Initial prompt
            reward: Initial reward
        """
        self.prompts_history.append(prompt)
        self.reward_history.append(reward)

    def step(self, retries: int = 3) -> str:
        """
        Performing a step of optimization - generating a new prompt.
        
        Args:
            retries: Number of attempts at unsuccessful generation
            
        Returns:
            New optimized prompt
        """
        if self._state != 0:
            raise ValueError("Expected to get reward before next step")
        
        if retries <= 0:
            logger.error("All attempts to generate prompt have been exhausted!")
            self.prompts_history.append(self.prompts_history[-1])
            self._state = 1
            return self.prompts_history[-1]
        
        # Forming history for transfer to LLM
        history_rows = []
        for reward, prompt in zip(self.reward_history, self.prompts_history):
            history_rows.append(
                f"{self.prompt_tokens[0]}{prompt}{self.prompt_tokens[1]}\n"
                f"{self.metric_name}: {reward}"
            )
            
        history = "\n\n".join(history_rows) + "\n\n" + self.task_prompt
        messages = [{"role": "system", "content": self.metaprompt + "\n\n" + history}]
        
        # Getting the answer from LLM
        response = self.llm_client.chat(messages)
        text = response.message.content
        
        # Extracting the prompt from the answer
        match = re.search(self.re_pattern, text, re.DOTALL)
        if match:
            new_prompt = match.group(1)
        else:
            logger.error("Ответ LLM не соответствует шаблону")
            return self.step(retries=retries-1)
        
        # Checking the validity of the prompt
        if self.check_fn is not None and not self.check_fn(new_prompt):
            logger.warning("Prompt did not pass the validity check")
            return self.step(retries=retries-1)
        
        self._state = 1
        self.prompts_history.append(new_prompt)
        
        return new_prompt

    def send_reward(self, reward: Union[float, int]) -> None:
        """
        Sending the reward for the last generated prompt.
        
        Args:
            reward: Value of the reward
            
        Raises:
            ValueError: If called in the wrong state
        """
        if self._state != 1:
            raise ValueError("Expected to generate a prompt before sending a reward")
        
        self._state = 0
        self.reward_history.append(reward)
