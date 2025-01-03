from .schema import AgentConfig, IOAgentConfig, AgentContext
from src.llm_client import LLMClient
from typing import List, Dict, Callable, Hashable
from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self, config: AgentConfig):
        self.config = config
        self.next_agents: List[str] = []
        self.previous_agents: List[str] = []

    def add_next_agent(self, agent_id: str):
        self.next_agents.append(agent_id)

    def add_previous_agent(self, agent_id: str):
        self.previous_agents.append(agent_id)

    def process(self, context: AgentContext) -> (AgentContext, List[str]):
        context = self.handle(context)
        return context, self.next_agents

    @abstractmethod
    def handle(self, context: AgentContext) -> AgentContext:
        pass


class IOAgent(Agent):
    def __init__(self, config: IOAgentConfig, llm_client: LLMClient):
        super().__init__(config)
        self.llm_client = llm_client

    def _format_messages(self, context: AgentContext) -> List[Dict[str, str]]:
        formated_messages = []
        for message in self.config.messages:
            formated_messages.append({"role": message["role"], "content": message["content"].format(**context.data)})
        return formated_messages

    def handle(self, context: AgentContext) -> AgentContext:
        messages = self._format_messages(context)
        response = self.llm_client.chat(messages)
        context.data[self.config.agent_id] = response.message.content
        return context


class EchoAgent(Agent):
    def __init__(self, config: IOAgentConfig):
        super().__init__(config)

    def handle(self, context: AgentContext) -> AgentContext:
        return context


class AgentFactory(ABC):
    @staticmethod
    def get_agent(agent_config_dct: dict, kwargs: dict = None) -> Agent:
        classes: dict[Hashable, (Callable[..., object], Callable[..., object])] = {
            "IO": (IOAgent, IOAgentConfig),
            "Echo": (EchoAgent, AgentConfig),
        }
        classes_ = classes.get(agent_config_dct["agent_type"], None)
        if classes_ is None:
            raise ValueError(f"Agent type {agent_config_dct['agent_type']} not supported")

        agent_class, agent_config_class = classes_
        agent_kwargs = {}
        if agent_config_dct["agent_type"] == "IO":
            agent_kwargs["llm_client"] = kwargs.get("llm_client", None)
        agent = agent_class(agent_config_class(**agent_config_dct), **agent_kwargs)
        return agent

