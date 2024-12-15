import networkx as nx
import matplotlib.pyplot as plt
from networkx import DiGraph, topological_sort
from typing import List, Dict
from .agent import Agent, AgentFactory
from .schema import AgentContext, PipelineAgentConfig
from src.llm_client import LLMClient


class PipelineGraphMixin:
    @staticmethod
    def validate_order(agents: dict, edges: List):
        graph = DiGraph()
        graph.add_nodes_from(agents.keys())
        graph.add_edges_from(edges)
        sorted_nodes = list(topological_sort(graph))
        for agent_id, agent in agents.items():
            neighbors = [neighbor.config.agent_id for neighbor in agent.next_agents]
            if sorted_nodes.index(agent_id) > max(sorted_nodes.index(neighbor_id) for neighbor_id in neighbors):
                raise Exception(f"Agent {agent_id} has next agents that are not in topological order.")

    @staticmethod
    def draw_graph(agents: dict, edges: List):
        graph = DiGraph()
        graph.add_nodes_from(agents.keys())
        graph.add_edges_from(edges)
        nx.draw(graph, with_labels=True)
        plt.show()

    @classmethod
    def sort_edges(cls, agents: dict, edges: List) -> List:
        graph = DiGraph()
        graph.add_nodes_from(agents.keys())
        graph.add_edges_from(edges)
        sorted_edges = [(u, v) for u in topological_sort(graph) for v in graph.successors(u)]
        for agent_id, agent in agents.items():
            agent.next_agents = [agents[neighbor_id] for (_, neighbor_id) in sorted_edges if neighbor_id == agent_id]
            agent.previous_agents = [agents[neighbor_id] for (neighbor_id, _) in sorted_edges if neighbor_id == agent_id]
        return sorted_edges


class Pipeline(PipelineGraphMixin):
    def __init__(self, config: PipelineAgentConfig, llm_client: LLMClient):
        self.config = config
        self.llm_client = llm_client
        self.agents = {}
        self.context = None
        self.output_agent_id = None

    def _from_config(self, config: PipelineAgentConfig):
        for agent_config_dct in config.agent_configs:
            agent = AgentFactory.get_agent(agent_config_dct, {"llm_client": self.llm_client})
            self.agents[agent.config.agent_id] = agent

        self.validate_order(agents, self.config.edges)
        
        for edge in config.edges:
            source_agent = self.agents[edge[0]]
            target_agent = self.agents[edge[1]]
            source_agent.add_next(target_agent)
            target_agent.add_previous(source_agent)

    def get_output(self):
        return self.context[output_agent_id]

    def process(self, input_text: str) -> AgentContext:
        self.context = AgentContext(input_text={"input": input_text})
        self.agents[0].process(self.context)
        return context


