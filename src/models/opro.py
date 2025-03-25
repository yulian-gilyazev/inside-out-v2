import re
from src.utils.logger import Logger
from src.llm_client import LLMClient
from typing import Callable, Optional, Tuple, List, Union, Any, Literal


class OPROMemory:
    """
    Memory for OPRO.
    """

    def __init__(self, strategy: Literal["last", "all", "ascending_subsequence"] = "last", n_last: int = 10):
        self.prompts_history: List[str] = []
        self.rewards_history: List[float] = []
        self.strategy = strategy
        self.n_last = n_last

    def add_prompt(self, prompt: str):
        self.prompts_history.append(prompt)
    
    def add_reward(self, reward: float):
        self.rewards_history.append(reward)
    
    def get_history(self) -> List[Tuple[str, float]]:
        # Last n prompts
        if self.strategy == "last":
            return list(zip(self.prompts_history[-self.n_last:], self.rewards_history[-self.n_last:]))
        # All history
        elif self.strategy == "all":
            return list(zip(self.prompts_history, self.rewards_history))
        # Greedy ascending subsequence strategy
        elif self.strategy == "ascending_subsequence":
            history = []
            for i in range(len(self.prompts_history)):
                if not history:
                    history.append((self.prompts_history[i], self.rewards_history[i]))
                    continue
                if self.rewards_history[i] > history[-1][1]:
                    history.append((self.prompts_history[i], self.rewards_history[i]))
            return history[-self.n_last:]
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")
        
    def get_last_prompt(self) -> str:
        return self.prompts_history[-1]
    
    def get_last_reward(self) -> float:
        return self.rewards_history[-1]
        
    def clear(self):
        self.prompts_history = []
        self.rewards_history = []
        

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
            prompt_tokens: Tuple[str, str] = ("<TEXT>", "</TEXT>"), 
            memory_strategy: Literal["last", "all"] = "all",
            memory_n_last: int = 10,
            logger: Logger = None
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
            memory_strategy: Strategy for memory (last, all, ascending_subsequence). 
                If last, only last n prompts are kept in memory,
                if all, all prompts are kept in memory, 
                if ascending_subsequence, only the prompts with ascending rewards are kept in memory
            memory_n_last: Number of last prompts to keep in memory
            logger: Logger
        """
        self.llm_client = llm_client
        self.memory = OPROMemory(strategy=memory_strategy, n_last=memory_n_last)
        self.metric_name = metric_name
        self.metaprompt = metaprompt
        self.task_prompt = task_prompt
        self.check_fn = check_fn
        self.prompt_tokens = prompt_tokens
        self.re_pattern = rf"{prompt_tokens[0]}(.*?){prompt_tokens[1]}"
        self.logger = logger
        # State: 0 - waiting for step, 1 - waiting for reward
        self._state = 0

    def initialize(self, prompt: str, reward: Union[float, int]) -> None:
        """
        Initialize the history of prompts and rewards.
        
        Args:
            prompt: Initial prompt
            reward: Initial reward
        """
        self.memory.add_prompt(prompt)
        self.memory.add_reward(reward)

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
            if self.logger is not None:
                self.logger.error("All attempts to generate prompt have been exhausted!")
            self.memory.add_prompt(self.memory.get_last_prompt())
            self._state = 1
            return self.memory.get_last_prompt()
        
        # Forming history for transfer to LLM
        history_rows = []
        for reward, prompt in self.memory.get_history():
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
            if self.logger is not None:
                self.logger.error("LLM response does not match the template")
                self.logger.info(text)
            return self.step(retries=retries-1)
        
        # Checking the validity of the prompt
        if self.check_fn is not None and not self.check_fn(new_prompt):
            if self.logger is not None:
                self.logger.warning("Prompt did not pass the validity check")
            return self.step(retries=retries-1)
        
        self._state = 1
        self.memory.add_prompt(new_prompt)
        
        return new_prompt

    def send_reward(self, reward: Union[float, int]) -> None:
        """
        Sending the reward for the last generated prompt.
        
        Args:
            reward: Value of the reward
        """
        if self._state != 1:
            raise ValueError("Expected to generate a prompt before sending a reward")
        
        self._state = 0
        self.memory.add_reward(reward)
