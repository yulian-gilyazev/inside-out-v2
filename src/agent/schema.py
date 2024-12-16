from typing import List, Dict, Optional, Callable, Any

from dataclasses import dataclass
from abc import ABC


@dataclass
class AgentContext(ABC):
    data: Dict[str, str]

    def get_value(self, name: str) -> str:
        return self.data.get(name, None)


@dataclass
class AgentConfig(ABC):
    agent_type: str
    agent_id: str


@dataclass
class IOAgentConfig(AgentConfig):
    messages: List[Dict[str, str]]


@dataclass
class PipelineAgentConfig(ABC):
    agent_configs: Dict[str, Dict[str, Any]]
    edges: List[List[str]]
    input_id: str
    output_id: str
