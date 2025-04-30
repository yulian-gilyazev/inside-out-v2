import os
from typing import Dict

from .schema import PipelineAgentConfig


class PipelineAgentConfigRegistry:
    def __init__(self):
        self.configs: Dict[str, PipelineAgentConfig] = {}

    def add_config(self, name, config: PipelineAgentConfig):
        self.configs[name] = config

    def get_config(self, name: str) -> PipelineAgentConfig:
        return self.configs[name]


def load_prompt(path: str) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, path), "r") as f:
        return f.read()


registry = PipelineAgentConfigRegistry()

system_prompt = load_prompt("agent_prompts/system_prompt.txt")


def get_inside_out_erc_config(is_extended: bool) -> PipelineAgentConfig:
    """
    Get the inside-out ERC config for the given emotions set.
    """
    if is_extended:
        emotions_list_str = load_prompt("agent_prompts/extended_emotions_list.txt")
        emotion_agent_prompt = load_prompt("agent_prompts/inside_out_extended_emotion_estimation.txt")
        aggregator_prompt = load_prompt("agent_prompts/inside_out_extended_aggregator.txt")
    else:
        emotions_list_str = load_prompt("agent_prompts/emotions_list.txt")
        emotion_agent_prompt = load_prompt("agent_prompts/inside_out_emotion_estimation.txt")
        aggregator_prompt = load_prompt("agent_prompts/inside_out_aggregator.txt")
    n_emotions = len(emotions_list_str.split(", "))
    return PipelineAgentConfig(
        agent_configs=[
            {
                "agent_type": "Echo",
                "agent_id": "input",
            },
            {
                "agent_type": "IO",
                "agent_id": "anger_agent",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt + "\n" + emotion_agent_prompt.format(emotion="Anger", emotions_list=emotions_list_str, n_emotions=n_emotions)
                    },
                    {"role": "user", "content": "Classify the emotion of speaker (A) in the following dialogue:\n\n{input}."},
                ]
            },
            {
                "agent_type": "IO",
                "agent_id": "disgust_agent",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt + "\n" + emotion_agent_prompt.format(emotion="Disgust", emotions_list=emotions_list_str, n_emotions=n_emotions)
                    },
                    {"role": "user", "content": "Classify the emotion of speaker (A) in the following dialogue:\n\n{input}."},
                ]
            },
            {
                "agent_type": "IO",
                "agent_id": "fear_agent",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt + "\n" + emotion_agent_prompt.format(emotion="Fear", emotions_list=emotions_list_str, n_emotions=n_emotions)
                    },
                    {"role": "user", "content": "Classify the emotion of speaker (A) in the following dialogue:\n\n{input}."},
                ]
            },
            {
                "agent_type": "IO",
                "agent_id": "happiness_agent",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt + "\n" + emotion_agent_prompt.format(emotion="Happiness", emotions_list=emotions_list_str, n_emotions=n_emotions)
                    },
                    {"role": "user", "content": "Classify the emotion of speaker (A) in the following dialogue:\n\n{input}."},
                ]
            },
            {
                "agent_type": "IO",
                "agent_id": "sadness_agent",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt + "\n" + emotion_agent_prompt.format(emotion="Sadness", emotions_list=emotions_list_str, n_emotions=n_emotions)
                    },
                    {"role": "user", "content": "Classify the emotion of speaker (A) in the following dialogue:\n\n{input}."},
                ]
            },
            {
                "agent_type": "IO",
                "agent_id": "aggregator",
                "messages": [
                    {"role": "system",
                     "content": system_prompt + "\n" + aggregator_prompt.format(emotions_list=emotions_list_str, n_emotions=n_emotions)
                     },
                    {"role": "user",
                     "content": "Classify the emotion of speaker (A) in the following dialogue using the following agent responses\n\nAgent responses:\n* Anger agent: {anger_agent}\n* Disgust agent: {disgust_agent}\n* Fear agent: {fear_agent}\n* Happiness agent: {happiness_agent}\n* Sadness agent: {sadness_agent}\n\n\nDialogue:\n{input}"
                     }
                ]
            },

        ],
        edges=[("input", "anger_agent"), ("input", "disgust_agent"), ("input", "fear_agent"), ("input", "happiness_agent"),
               ("input", "sadness_agent"), ("anger_agent", "aggregator"), ("disgust_agent", "aggregator"),
               ("fear_agent", "aggregator"), ("happiness_agent", "aggregator"), ("sadness_agent", "aggregator")],
        input_id="input",
        output_id="aggregator",
    )


registry.add_config(
    "inside-out-erc",
    get_inside_out_erc_config(is_extended=False)
)

registry.add_config(
    "inside-out-erc-extended-emotions",
    get_inside_out_erc_config(is_extended=True)
)


def get_baseline_erc_config(is_extended: bool) -> PipelineAgentConfig:
    if is_extended:
        emotions_list_str = load_prompt("agent_prompts/extended_emotions_list.txt")
        n_emotions = len(emotions_list_str.split("\n"))
        prompt = load_prompt("agent_prompts/baseline_erc_extended.txt")
    else:
        emotions_list_str = load_prompt("agent_prompts/emotions_list.txt")
        n_emotions = len(emotions_list_str.split("\n"))
        prompt = load_prompt("agent_prompts/baseline_erc.txt")
    return PipelineAgentConfig(
        agent_configs=[
            {
                "agent_type": "IO",
                "agent_id": "erc",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt + "\n" + prompt.format(n_emotions=n_emotions, emotions_list=emotions_list_str)
                    },
                    {
                        "role": "user",
                        "content": "Dialogue:\n{input}."
                    }
                ]
            },
        ],
        edges=[],
        input_id="erc",
        output_id="erc",
    )

registry.add_config(
    "baseline-erc",
    get_baseline_erc_config(is_extended=False)
)

registry.add_config(
    "baseline-erc-extended-emotions",
    get_baseline_erc_config(is_extended=True)
)


def get_self_consistency_erc_config(is_extended: bool, n_agents: int) -> PipelineAgentConfig:
    if is_extended:
        emotions_list_str = load_prompt("agent_prompts/extended_emotions_list.txt")
        n_emotions = len(emotions_list_str.split("\n"))
        prompt = load_prompt("agent_prompts/baseline_erc_extended.txt")
        aggregator_prompt = load_prompt("agent_prompts/inside_out_extended_aggregator.txt")
    else:
        emotions_list_str = load_prompt("agent_prompts/emotions_list.txt")
        n_emotions = len(emotions_list_str.split("\n"))
        prompt = load_prompt("agent_prompts/baseline_erc.txt")
        aggregator_prompt = load_prompt("agent_prompts/inside_out_aggregator.txt")

    agent_configs = [
        {
            "agent_type": "Echo",
            "agent_id": "input",
        },
    ]
    for i in range(n_agents):
        agent_configs.append({
            "agent_type": "IO",
            "agent_id": f"agent_{i + 1}",
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt + "\n" + prompt.format(n_emotions=n_emotions, emotions_list=emotions_list_str)
                },
                {
                    "role": "user",
                    "content": "Dialogue:\n{input}."
                }
            ]
        })
    agent_configs.append({
            "agent_type": "IO",
            "agent_id": "aggregator",
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt + "\n" + aggregator_prompt.format(emotions_list=emotions_list_str, n_emotions=n_emotions)
                },
                {
                    "role": "user",
                    "content": "Dialogue:\n{input}\nAgent responses:\n* " + "\n* ".join([f"agent_{i + 1}" for i in range(n_agents)]) + "\n"
                }
            ]
        })
    edges = [("input", f"agent_{i + 1}") for i in range(n_agents)] + [(f"agent_{i + 1}", "aggregator") for i in range(n_agents)]
    return PipelineAgentConfig(
        agent_configs=agent_configs,
        edges=edges,
        input_id="input",
        output_id="aggregator",
    )

registry.add_config(
    "self-consistency-erc-3-agents",
    get_self_consistency_erc_config(is_extended=False, n_agents=3)
)

registry.add_config(
    "self-consistency-erc-5-agents",
    get_self_consistency_erc_config(is_extended=False, n_agents=5)
)

registry.add_config(
    "self-consistency-erc-extended-emotions-3-agents",
    get_self_consistency_erc_config(is_extended=True, n_agents=3)
)

registry.add_config(
    "self-consistency-erc-extended-emotions-5-agents",
    get_self_consistency_erc_config(is_extended=True, n_agents=5)
)