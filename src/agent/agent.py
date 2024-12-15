from dataclasses import dataclass
from .schema import AgentConfig, IOAgentConfig, AgentContext
from typing import List, Dict, Optional, Callable, Any
from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self, config: AgentConfig):
        self.config = config
        self.next_agents: List[Agent] = []
        self.previous_agents: List[Agent] = []

    def add_next_agent(self, agent: Agent):
        self.next_agents.append(agent)

    def add_previous_agent(self, agent: Agent):
        self.previous_agents.append(agent)

    def process(self, context: AgentContext) -> AgentContext:
        context = self.handle(context)
        for agent in self.next_agents:
            context = agent.process(context)
        return context

    @abstractmethod
    def handle(self, context: AgentContext) -> AgentContext:
        pass


class IOAgent(Agent):
    def __init__(self, config: IOAgentConfig, llm_client, messages: List[Dict[str, str]]):
        super().__init__(config)
        self.llm_client = llm_client
        self.messages = messages

    def _format_messages(self, context: AgentContext) -> List[Dict[str, str]]:
        formated_messages = []
        for message in self.messages:
            formated_messages.append({"role": message["role"], "content": message["content"].format(context.data)})
        return formated_messages

    def handle(self, context: AgentContext) -> AgentContext:
        messages = self._format_messages(context)
        response = self.llm_client.chat(messages)
        context.data[self.agent_config.agent_id] = response.message.content
        return context


class AgentFactory:
    @staticmethod
    def get_agent(agent_config_dct: dict, kwargs: dict = None) -> Agent:
        classes: dict[Hashable, (Callable[..., object], Callable[..., object])] = {
            "IO": (IOAgent, IOAgentConfig),
        }
        classes_ = classes.get(agent_config_dct["agent_type"], None)
        if classes_ is None:
            raise ValueError(f"Agent type {agent_type} not supported")

        agent_class, agent_config_class = classes_
        agent_kwargs = {}
        if agent_config_dct["agent_type"] == "IO":
            agent_kwargs["llm_clien"] = kwargs.get(agent_config_dct["llm_client"], None)
        agent = agent_class(agent_config_class(agent_config_dct), **agent_kwargs)
        return agent

