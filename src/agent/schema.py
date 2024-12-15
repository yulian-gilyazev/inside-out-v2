from typing import List, Dict, Optional, Callable, Any

from abc import ABC


@dataclass
class AgentContext(ABC):
    data: Dict[str, str]
    metadata: Dict[str, Any] = None


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
    edges: List[List[str, str]]
    input_id: str
    output_id: str
