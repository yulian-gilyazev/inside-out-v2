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
emotion_agent_prompt = load_prompt("agent_prompts/inside_out_emotion_estimation.txt")


emotions_list_str = load_prompt("agent_prompts/emotions_list.txt")
n_emotions = len(emotions_list_str.split("\n"))

registry.add_config(
    "inside-out-erc",
    PipelineAgentConfig(
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
                        "content": system_prompt + "\n" + load_prompt("agent_prompts/inside_out_emotion_estimation.txt").format(emotion="Anger", emotions_list=emotions_list_str, n_emotions=n_emotions)
                    },
                    {"role": "user", "content": "Dialogue:\n{input}."},
                ]
            },
            {
                "agent_type": "IO",
                "agent_id": "disgust_agent",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt + "\n" + load_prompt("agent_prompts/inside_out_emotion_estimation.txt").format(emotion="Disgust", emotions_list=emotions_list_str, n_emotions=n_emotions)
                    },
                    {"role": "user", "content": "Dialogue:\n{input}."},
                ]
            },
            {
                "agent_type": "IO",
                "agent_id": "fear_agent",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt + "\n" + load_prompt("agent_prompts/inside_out_emotion_estimation.txt").format(emotion="Fear", emotions_list=emotions_list_str, n_emotions=n_emotions)
                    },
                    {"role": "user", "content": "Dialogue:\n{input}."},
                ]
            },
            {
                "agent_type": "IO",
                "agent_id": "happiness_agent",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt + "\n" + load_prompt("agent_prompts/inside_out_emotion_estimation.txt").format(emotion="Happiness", emotions_list=emotions_list_str, n_emotions=n_emotions)
                    },
                    {"role": "user", "content": "Dialogue:\n{input}."},
                ]
            },
            {
                "agent_type": "IO",
                "agent_id": "sadness_agent",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt + "\n" + load_prompt("agent_prompts/inside_out_emotion_estimation.txt").format(emotion="Sadness", emotions_list=emotions_list_str, n_emotions=n_emotions)
                    },
                    {"role": "user", "content": "Dialogue:\n{input}."},
                ]
            },
            {
                "agent_type": "IO",
                "agent_id": "aggregator",
                "messages": [
                    {"role": "system",
                     "content": system_prompt + "\n" + load_prompt("agent_prompts/inside_out_aggregator.txt").format(emotions_list=emotions_list_str, n_emotions=n_emotions)
                     },
                    {"role": "user",
                     "content": "Dialogue:\n{input}\nAgent responses:\n* {anger_agent}\n* {disgust_agent}\n* {fear_agent}\n* {happiness_agent}\n* {sadness_agent}"
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
)


registry.add_config(
    "baseline-erc",
    PipelineAgentConfig(
        agent_configs=[
            {
                "agent_type": "IO",
                "agent_id": "erc",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt + "\n" + load_prompt("agent_prompts/baseline_erc.txt")
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
)


registry.add_config(
    "self-consistency-erc",
    PipelineAgentConfig(
        agent_configs=[
            {
                "agent_type": "Echo",
                "agent_id": "input",
            },
            {
                "agent_type": "IO",
                "agent_id": "first",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt + "\n" + load_prompt("agent_prompts/baseline_erc.txt")
                    },
                    {
                        "role": "user",
                        "content": "Dialogue:\n{input}."
                    }
                ]
            },
            {
                "agent_type": "IO",
                "agent_id": "second",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt + "\n" + load_prompt("agent_prompts/baseline_erc.txt")
                    },
                    {
                        "role": "user",
                        "content": "Dialogue:\n{input}."
                    }
                ]
            },
            {
                "agent_type": "IO",
                "agent_id": "third",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt + "\n" + load_prompt("agent_prompts/baseline_erc.txt")
                    },
                    {
                        "role": "user",
                        "content": "Dialogue:\n{input}."
                    }
                ]
            },
            {
                "agent_type": "IO",
                "agent_id": "aggregator",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt + "\n" + load_prompt("agent_prompts/inside_out_aggregator.txt")
                    },
                    {
                        "role": "user",
                        "content": "Dialogue:\n{input}\nAgent responses:\n* {first}\n* {second}\n* {third}\n"
                    }
                ]
            },
        ],
        edges=[("input", "first"), ("input", "second"), ("input", "third"),
               ("first", "aggregator"), ("second", "aggregator"), ("third", "aggregator")],
        input_id="input",
        output_id="aggregator",
    )
)


extended_emotions_list_str = load_prompt("agent_prompts/extended_emotions_list.txt")
n_extended_emotions = len(extended_emotions_list_str.split("\n"))

registry.add_config(
    "inside-out-erc-extended-emotions",
    PipelineAgentConfig(
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
                        "content": system_prompt + "\n" + emotion_agent_prompt.format(emotion="Anger", emotions_list=extended_emotions_list_str, n_emotions=n_extended_emotions)
                    },
                    {"role": "user", "content": "Dialogue:\n{input}."},
                ]
            },
            {
                "agent_type": "IO",
                "agent_id": "disgust_agent",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt + "\n" + emotion_agent_prompt.format(emotion="Disgust", emotions_list=extended_emotions_list_str, n_emotions=n_extended_emotions)
                    },
                    {"role": "user", "content": "Dialogue:\n{input}."},
                ]
            },
            {
                "agent_type": "IO",
                "agent_id": "fear_agent",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt + "\n" + emotion_agent_prompt.format(emotion="Fear", emotions_list=extended_emotions_list_str, n_emotions=n_extended_emotions)
                    },
                    {"role": "user", "content": "Dialogue:\n{input}."},
                ]
            },
            {
                "agent_type": "IO",
                "agent_id": "happiness_agent",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt + "\n" + emotion_agent_prompt.format(emotion="Happiness", emotions_list=extended_emotions_list_str, n_emotions=n_extended_emotions)
                    },
                    {"role": "user", "content": "Dialogue:\n{input}."},
                ]
            },
            {
                "agent_type": "IO",
                "agent_id": "sadness_agent",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt + "\n" + emotion_agent_prompt.format(emotion="Sadness", emotions_list=extended_emotions_list_str, n_emotions=n_extended_emotions)
                    },
                    {"role": "user", "content": "Dialogue:\n{input}."},
                ]
            },
            {
                "agent_type": "IO",
                "agent_id": "aggregator",
                "messages": [
                    {"role": "system",
                     "content": system_prompt + "\n" + load_prompt("agent_prompts/inside_out_aggregator.txt").format(emotions_list=emotions_list_str, n_emotions=n_emotions)
                     },
                    {"role": "user",
                     "content": "Dialogue:\n{input}\nAgent responses:\n* {anger_agent}\n* {disgust_agent}\n* {fear_agent}\n* {happiness_agent}\n* {sadness_agent}"
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
)