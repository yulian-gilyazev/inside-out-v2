import argparse
from dataclasses import dataclass
import json
import os
from typing import List, Dict, Tuple, Any
import re
import copy
from src.agent import PipelineAgentConfig, AgentConfig, IOAgentConfig, Agent, AgentContext, AgentFactory, Pipeline
from src.agent import IOAgent
from src.llm_client import LLMClient
from src.schema.llm_config import LLMConfig
from src.utils.data import SyntheticEmotionDataset, EmpatheticDialoguesDataset, split_dataset
from tqdm.auto import tqdm
from loguru import logger


"""
python3 -m src.scripts.experiments.inside-out-alt-topology-v2 --dataset 'synthetic' --dataset_path 'data/synthetic_dialogues/v2' --out_path 'data/inside_out_alt_topology_v2_debug.json'

python3 -m src.scripts.experiments.inside-out-alt-topology-v2 --dataset 'empatheticdialogues' --dataset_path 'data/empatheticdialogues' --part 'test' --out_path 'data/empatheticdialogues_test_inside_out_alt_topology_v2_exp1.json'
"""


class EmotionParserAgent(Agent):
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.re_pattern = rf"{config.emotion_tokens[0]}(.*?){config.emotion_tokens[1]}"

    def handle(self, context: AgentContext) -> AgentContext:
        result = []
        for match in re.finditer(self.re_pattern, context.get_value(self.config.input_id), re.DOTALL):
            result.append(match.group(1))
        return result
    

class MultipleIOFromTemplateAgent(Agent):
    """
    Perform multiple IO operations from a template messages.
    """
    def __init__(self, config: AgentConfig, llm_client: LLMClient):
        super().__init__(config)
        curr_config_dct = {"agent_type": "IO", "agent_id": f"io_agent_{self.config.agent_id}", "messages": config.messages}
        self.io_agent = AgentFactory.get_agent(curr_config_dct, kwargs={"llm_client": llm_client})
       
    def handle(self, context: AgentContext) -> AgentContext:
        results = []
        for item in context.get_value(self.config.input_id):
            curr_context = copy.deepcopy(context)
            curr_context.data[self.config.input_id] = item
            results.append(self.io_agent.handle(curr_context))
        return results
    

class MultipleIOFromTemplateDebateAgent(MultipleIOFromTemplateAgent):
    """
    Perform debate between multiple IO agents.
    """
    def handle(self, context: AgentContext) -> AgentContext:
        results = []
        previous_results = []
        for emotion, item in zip(context.get_value(self.config.previous_results_id), context.get_value(self.config.input_id)):
            previous_results.append(f"Emotion agent: {emotion}\t response: {item}")
        
        previous_results_str = "\n".join(previous_results)

        for item in context.get_value(self.config.input_id):
            curr_context = copy.deepcopy(context)
            curr_context.data[self.config.input_id] = item
            curr_context.data[self.config.previous_results_concated_id] = previous_results_str
            results.append(self.io_agent.handle(curr_context))
        return results


class ListConcatenatorAgent(Agent):
    def __init__(self, config: AgentConfig):
        super().__init__(config)
    
    def handle(self, context: AgentContext) -> AgentContext:
        return self.config.separator.join(context.get_value(self.config.input_id))

@dataclass
class EmotionParserAgentConfig(AgentConfig):
    emotion_tokens: Tuple[str, str]
    input_id: str

@dataclass
class MultipleIOFromTemplateAgentConfig(AgentConfig):
    input_id: str
    messages: List[Dict[str, Any]]


@dataclass
class MultipleIOFromTemplateDebateAgentConfig(AgentConfig):
    input_id: str
    messages: List[Dict[str, Any]]
    previous_results_id: str
    previous_results_concated_id: str

@dataclass
class ListConcatenatorAgentConfig(AgentConfig):
    separator: str
    input_id: str


AgentFactory.add_agent(
    "EmotionParser", 
    EmotionParserAgent,
    EmotionParserAgentConfig, 
    use_llm=False
)

AgentFactory.add_agent(
    "MultipleIOFromTemplate", 
    MultipleIOFromTemplateAgent,
    MultipleIOFromTemplateAgentConfig,
    use_llm=True
)

AgentFactory.add_agent(
    "MultipleIOFromTemplateDebate", 
    MultipleIOFromTemplateDebateAgent,
    MultipleIOFromTemplateDebateAgentConfig,
    use_llm=True
)
AgentFactory.add_agent(
    "ListConcatenator", 
    ListConcatenatorAgent,
    ListConcatenatorAgentConfig, 
    use_llm=False
)


system_prompt = """You are a highly advanced language model.
Carefully heed the user's instructions."""


def get_inside_out_exp_pipeline_cfg():
    pipeline_config = PipelineAgentConfig(
        agent_configs=[
            {
                "agent_type": "Echo",
                "agent_id": "input",
            },
            {
                "agent_type": "IO",
                "agent_id": "emotions_generator",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt + "\n\n" + """Your task is to generate a set of emotional states for a dialogue evaluator. The evaluator's goal is to accurately identify the emotion of the first speaker in a given dialogue.

The evaluator will be provided with both the dialogue and the emotional states you generate. These emotional states will guide the evaluator in determining the first speaker's emotion.

Key Points:
- The accuracy of the evaluator's assessment is directly influenced by the emotional states you generate. Some emotions will aid in correctly identifying the first speaker's emotion, while others may complicate the task.
- Generate emotions that are likely to be experienced by the evaluator, making the task easier. Different agents will experience different emotions, and the responses from all agents will be aggregated in the emotion recognition task.
- For instance, in a conflict dialogue, it is improbable that both participants will be happy. Avoid generating emotions that could complicate the evaluator's task.

Format Guidelines:
- Enclose each emotion in <EMOTION>Emotion</EMOTION> tags.
- Use only the 5 basic emotions from Ekman's list: Anger, Disgust, Fear, Happiness, Sadness.
- You may generate combinations of two emotions, e.g., <EMOTION>Sadness and Disgust</EMOTION>.
- Avoid repeating the same emotion or combination of emotions.
- Generate between 2 to 5 distinct emotional states, depending on the dialogue's complexity.
- Example output: <EMOTION>Anger</EMOTION> <EMOTION>Fear and Anger</EMOTION> <EMOTION>Sadness</EMOTION>

Keep in mind that your selection of emotions will have a significant impact on the evaluator's performance.
                        """
                    },
                    {"role": "user", "content": "Dialogue:\n{input}."},
                ]
            },
            {
                "agent_type": "EmotionParser",
                "agent_id": "emotion_parser",
                "input_id": "emotions_generator",
                "emotion_tokens": ("<EMOTION>", "</EMOTION>"),
            },

            {
                "agent_type": "MultipleIOFromTemplate",
                "agent_id": "inside_out_agents",
                "input_id": "emotion_parser",
                "messages": [
                    {"role": "system", "content": system_prompt + "\n" + """You feel {emotion_parser}. Act based on what emotion you are experiencing.
You need to assess emotion of the first (A) interlocutor in the dialogue, estimate your confidence and give reasoning for your answer.
Your answer should consist of an emotion and an assessment of the level of confidence in it in the range from 0 to 1.
To select emotions, use Ekman's classification into 5 main emotions - Anger, Disgust, Fear, Happiness, Sadness. 
Separate the emotion and the response using a semicolon.
Response example:
`Anger; 0.7`"""},
                     {"role": "user", "content": "Dialogue:\n{input}."}
                ]
            },
            {
                "agent_type": "MultipleIOFromTemplateDebate",
                "agent_id": "inside_out_agents_debate_round1",
                "input_id": "emotion_parser",
                "previous_results_id": "inside_out_agents",
                "previous_results_concated_id": "inside_out_agents_debate_round1_concated",
                "messages": [
                    {"role": "system", "content": system_prompt + "\n" + """You feel {emotion_parser}. Act based on what emotion you are experiencing.
You need to assess emotion of the first (A) interlocutor in the dialogue, estimate your confidence and give reasoning for your answer.
Your answer should consist of an emotion and an assessment of the level of confidence in it in the range from 0 to 1.
To select emotions, use Ekman's classification into 5 main emotions - Anger, Disgust, Fear, Happiness, Sadness. 
Separate the emotion and the response using a semicolon.
Response example:
`Anger; 0.7`"""},
                     {"role": "user", "content": """You will also be given the responses from other emotional agents and your own response from the previous round of debate. This information will help you give your answer more confidently.
Using the solutions from other emotional agents (each agent has the same task as you, but feels different emotions) and your own response from the previous round of debate as additional information, give a response. 
Emotional agents responses:\n{inside_out_agents_debate_round1_concated}\n\n\n Dialogue:\n{input}."""}
                ]
            },
            {
                "agent_type": "ListConcatenator",
                "agent_id": "inside_out_concatenator",
                "input_id": "inside_out_agents_debate_round1",
                "separator": "\n* "
            },
            {
                "agent_type": "IO",
                "agent_id": "aggregator",
                "messages": [
                    {"role": "system",
                     "content": system_prompt + "\n" + """You have been given answers by several emotional agents, each of whom was interviewed to assess the emotional state of the first (A) interlocutor in the dialogue.
You are also given the dialogue itself.
Your task is to aggregate the responses of these agents and give your own based on the dialogue and the responses of the agents.
Your answer should consist of an emotion and an assessment of the level of confidence in it in the range from 0 to 1.
To select emotions, use Ekman's classification into 5 main emotions - Anger, Disgust, Fear, Happiness, Sadness.
Separate the emotion and the response using a semicolon.
Response example:
`Anger; 0.7`
The same format is followed for agent responses."""
                     },
                    {"role": "user",
                     "content": "Dialogue:\n{input}\nAgent responses:\n* {inside_out_concatenator}"
                     }
                ]
            },

        ],
        edges=[("input", "emotions_generator"), 
               ("emotions_generator", "emotion_parser"), 
               ("emotion_parser", "inside_out_agents"), 
               ("inside_out_agents", "inside_out_agents_debate_round1"),
               ("inside_out_agents_debate_round1", "inside_out_concatenator"), 
               ("inside_out_concatenator", "aggregator")],
        input_id="input",
        output_id="aggregator",
    )
    return pipeline_config


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=["synthetic", "empatheticdialogues"], help='Dataset to use')
    parser.add_argument('--dataset_path', type=str, help='Path to dataset')
    parser.add_argument('--part', type=str, choices=["train", "dev", "test"], required=False, help='Part of dataset to use')
    parser.add_argument('--llm_config_path', type=str,
                        default="configs/llm_generation/gpt_4o_mini_config.json", help='Path to llm config')
    parser.add_argument('--out_path', type=str, help='Path where scenarios will be saved')
    args = parser.parse_args()
    if args.dataset == "empatheticdialogues":
        assert args.part is not None, "Part must be specified for synthetic dataset"
    return args


def main():
    args = parse_arguments()

    with open(args.llm_config_path, "r") as f:
        config_dct = json.load(f)
    llm_client = LLMClient(LLMConfig.from_dict(config_dct))

    inside_out_pipeline_config = get_inside_out_exp_pipeline_cfg()

    pipeline = Pipeline(inside_out_pipeline_config, llm_client)
    if args.dataset == "synthetic":
        dialogues_path = os.path.join(args.dataset_path, "dialogues.json")
        scenarios_path = os.path.join(args.dataset_path, "scenarios.json")
        dset = SyntheticEmotionDataset(dialogues_path, scenarios_path)
        dset, _ = split_dataset(dset, 200)
    elif args.dataset == "empatheticdialogues":
        dset = EmpatheticDialoguesDataset(args.dataset_path, args.part)

    logger.info(f"Start inference on {len(dset)} dialogues")
    result = []
    for idx in tqdm(range(len(dset))):
        item = dset[idx]

        context = AgentContext(data={"input": item.format_dialogue()})

        context = pipeline.process(context)
        predicted = context.get_value(inside_out_pipeline_config.output_id)
        result.append({"id": item.id, "empathy_label": item.empathy_label, "prediction": predicted})

    with open(args.out_path, "w") as f:
        json.dump({"predictions": result}, f)
    logger.info(f"Saved results to {args.out_path}")

    logger.info(f"Completion tokens: {llm_client.get_output_tokens().sum()}")
    logger.info(f"Prompt tokens: {llm_client.get_input_tokens().sum()}")
    logger.info(f"Generation cost: {llm_client.get_generations_cost()}")


if __name__ == "__main__":
    main()