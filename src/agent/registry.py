import os

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
                        "content": system_prompt + "\n" + load_prompt("agent_prompts/inside_out_emotion_estimation.txt").format(emotion="Anger")
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
                        "content": system_prompt + "\n" + load_prompt("agent_prompts/inside_out_emotion_estimation.txt").format(emotion="Disgust")
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
                        "content": system_prompt + "\n" + load_prompt("agent_prompts/inside_out_emotion_estimation.txt").format(emotion="Fear")
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
                        "content": system_prompt + "\n" + load_prompt("agent_prompts/inside_out_emotion_estimation.txt").format(emotion="Happiness")
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
                        "content": system_prompt + "\n" + load_prompt("agent_prompts/inside_out_emotion_estimation.txt").format(emotion="Sadness")
                    },
                    {"role": "user", "content": "Dialogue:\n{input}."},
                ]
            },
            {
                "agent_type": "IO",
                "agent_id": "aggregator",
                "messages": [
                    {"role": "system",
                     "content": system_prompt + "\n" + load_prompt("agent_prompts/inside_out_aggregator.txt")
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
        input_id="input", 
        output_id="erc",
    )
)
