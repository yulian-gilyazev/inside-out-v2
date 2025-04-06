import argparse
import copy
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from src.agent import (Agent, AgentConfig, AgentContext, AgentFactory, IOAgent,
                       IOAgentConfig, Pipeline, PipelineAgentConfig)
from src.llm_client import LLMClient
from src.models.opro import OPRO
from src.schema.llm_config import LLMConfig
from src.scripts.experiments.agents_opro import (OptimizePipelineConfig,
                                                 optimize_pipeline)
from src.scripts.experiments.utils import accuracy, run_pipeline
from src.utils.data import (EmpatheticDialoguesDataset,
                            SyntheticEmotionDataset, split_dataset)
from src.utils.logger import Logger
from src.utils.prompts import (get_emotional_agent_debate_prompt,
                               get_emotions_generation_prompt,
                               get_inside_out_aggregator_prompt,
                               get_inside_out_emotinoal_prompt,
                               get_system_prompt)

"""
python3 -m src.scripts.experiments.inside-out-alt-topology-v2 --dataset 'synthetic' --dataset_path 'data/synthetic_dialogues/v2' --out_path 'data/inside_out_alt_topology_v2_debug.json'

python3 -m src.scripts.experiments.inside-out-alt-topology-v2 --dataset 'empatheticdialogues' --dataset_path 'data/empatheticdialogues' --part 'test' --out_path 'data/empatheticdialogues_test_inside_out_alt_topology_v2_exp1.json' --is_extended
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
        for emotion, item in zip(context.get_value(self.config.input_id),context.get_value(self.config.previous_results_id)):
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
        result = []
        for key, item in zip(context.get_value(self.config.key_id), context.get_value(self.config.input_id)):
            result.append(f"{key} agent: {item}")
        return self.config.separator.join(result)


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
    key_id: str


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


def get_inside_out_exp_pipeline_cfg(system_prompt: str,
                                    emotions_generation_prompt: str,
                                    emotional_agent_prompt: str,
                                    emotional_agent_debate_prompt: str,
                                    aggregator_prompt: str):
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
                        "content": system_prompt + "\n\n" + emotions_generation_prompt
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
                    {"role": "system", "content": system_prompt + "\n" + emotional_agent_prompt + "\nIn addition to evaluating the dialohue, provide an analysis of the dialogue and assumptions about the emotional state of the first interlocutor in the conversation."},
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
                    {"role": "system", "content": system_prompt + "\n" + emotional_agent_prompt + "\n\n" + emotional_agent_debate_prompt.replace("{prev_round_key}", "{inside_out_agents_debate_round1_concated}")},
                    {"role": "user", "content": "Dialogue:\n{input}."}
                ]
            },
            {
                "agent_type": "ListConcatenator",
                "agent_id": "inside_out_concatenator",
                "input_id": "inside_out_agents_debate_round1",
                "key_id": "emotion_parser",
                "separator": "\n* "
            },
            {
                "agent_type": "IO",
                "agent_id": "aggregator",
                "messages": [
                    {"role": "system",
                     "content": system_prompt + "\n" + aggregator_prompt
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

OPRO_METAPROMPT = """You are an AI assistant specializing in optimizing prompts for emotion classification.

## Task Description
Optimize an Emotion Recognition prompts for multi-agent pipeline for emotion classification tasks. The pipeline must accurately classify the emotion of the first (A) interlocutor in dialogues into one of the categories, along with a confidence score.

## Pipeline Specifications
- Powered by Large Language Models (LLMs)
- Leverages inter-agent communication between LLM calls for enhanced performance
- Implements inside-out idea, where the pipeline is composed of multiple agents, each of which is an LLM call
- First agent generates emotions
- Other groups of agents implement inside-out idea, where they are prompted to feel the emotion generated by the first agent and then to assess the emotion of the first interlocutor in the dialogue.
- Inside-out agents have debate between them to improve the accuracy of the emotion assessment.
- Last agent aggregates the results of the other agents and outputs the final result.

## Input
- Historical pipeline prompts in JSON format with accuracy metrics (scale: 0 to 1), in format:
```
<PROMPT>{prompts}</PROMPT> 
accuracy: {accuracy}
```
- Prompts should be in the same format as original prompts - they are given in dictionary format with some keys and corresponding prompts.
- Do not change keys, only prompts.

## Modification Scope
You may modify:
- Given prompts set
- IMPORTANT: Do not alter template variables (e.g., {emotion_parser}, {anger_agent}, {aggregator}) as they are essential for passing information between agents
- IMPORTANT: Do not change the number of prompts. You can only change prompts.

## Requirements
You must preserve:
- The idea of pipeline
- Key architectural components and their relationships

## Critical Guidelines
- Maintain the working structure of prompts - they should be able to be used in the same way as original prompts
- Prioritize accuracy metric improvements
- Analyze previous high-performing examples for insights
- Ensure valid and properly formatted output in the same format as original prompts

Return your optimized pipeline configuration within <PROMPT> tags in valid JSON format, designed to maximize classification accuracy. Before returning the configuration, you are allowed to give analysis of the previous prompts and the results and suggest changes. But stick to the format given in the description.
"""

OPRO_TASK_PROMPT = """Analyze previous prompts and create an optimized version that outperforms all prior examples in terms of quality and accuracy scores. Focus on refining agent interactions and prompt engineering to maximize emotion classification performance. Your response must contain ONLY the configuration JSON, enclosed within <PROMPT> and </PROMPT> tags. Do not include any explanations, comments, or additional text outside these tags."""



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=["synthetic", "empatheticdialogues"], help='Dataset to use')
    parser.add_argument('--dataset_path', type=str, help='Path to dataset')
    parser.add_argument('--part', type=str, choices=["train", "dev", "test"], required=False, help='Part of dataset to use')
    parser.add_argument('--llm_config_path', type=str,
                        default="configs/llm_generation/openai_gpt_4_config.json", help='Path to llm config')
    parser.add_argument('--train_size', type=int, default=300, help='Train size')
    parser.add_argument('--is_extended', action='store_true', help='Use extended dataset')
    parser.add_argument('--out_path', type=str, help='Path where scenarios will be saved')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for evaluation')
    parser.add_argument('--action', type=str, default="optimize", choices=["optimize", "evaluate"], help='Action to perform')

    args = parser.parse_args()
    if args.dataset == "empatheticdialogues":
        assert args.part is not None, "Part must be specified for synthetic dataset"
    return args

def main():
    args = parse_arguments()

    with open(args.llm_config_path, "r") as f:
        config_dct = json.load(f)
    llm_client = LLMClient(LLMConfig.from_dict(config_dct))

    emotions_generation_prompt = get_emotions_generation_prompt(is_extended=args.is_extended)
    emotional_agent_prompt = get_inside_out_emotinoal_prompt(is_extended=args.is_extended)
    aggregator_prompt = get_inside_out_aggregator_prompt(is_extended=args.is_extended)
    emotional_agent_debate_prompt = get_emotional_agent_debate_prompt()
    system_prompt = get_system_prompt()

    emotional_agent_prompt = emotional_agent_prompt.replace("{emotion}", "{emotion_parser}")

    if args.dataset == "synthetic":
        dialogues_path = os.path.join(args.dataset_path, "dialogues.json")
        scenarios_path = os.path.join(args.dataset_path, "scenarios.json")
        dset = SyntheticEmotionDataset(dialogues_path, scenarios_path)
        dset, train_dset = split_dataset(dset, 300)
        train_dset, _ = split_dataset(train_dset, args.train_size)
    elif args.dataset == "empatheticdialogues":
        dset = EmpatheticDialoguesDataset(args.dataset_path, args.part, extended=args.is_extended)
        dset, _ = split_dataset(dset, 100)
        train_dset = EmpatheticDialoguesDataset(args.dataset_path, "train", extended=args.is_extended)
        train_dset, _ = split_dataset(train_dset, args.train_size)

    if args.action == "optimize":
        with open(args.llm_prompt_searcher_config_path, "r") as f:
            config_dct = json.load(f)
        llm_prompt_searcher = LLMClient(LLMConfig.from_dict(config_dct))

        optimize_config = OptimizePipelineConfig(num_workers=args.num_workers)
        logger = Logger(
            group="inside-out-alt-topology-v2-prompt-optimization",
            run_name="run_empatheticdialogues_v2_04_03",
            tags=["inside-out-alt-topology-v2", "empatheticdialogues"],
            config=optimize_config.to_dict(),
            use_wandb=True,
        )

        prompts = {
            "system_prompt": system_prompt,
            "emotions_generation_prompt": emotions_generation_prompt,
            "emotional_agent_prompt": emotional_agent_prompt,
            "emotional_agent_debate_prompt": emotional_agent_debate_prompt,
            "aggregator_prompt": aggregator_prompt,
        }

        optimize_pipeline(
            llm_client, 
            llm_prompt_searcher,
            logger,
            optimize_config,
            train_dset,
            dset,
            OPRO_METAPROMPT,
            OPRO_TASK_PROMPT,
            opro_prompt_tokens=("<PROMPT>", "</PROMPT>"),
            prompts=prompts, 
            cfg_from_prompts_fn=get_inside_out_exp_pipeline_cfg,
            check_fn=None)
    
    else:
        assert args.out_path is not None, "Output path must be specified"
        inside_out_pipeline_config = get_inside_out_exp_pipeline_cfg(
            system_prompt=system_prompt,
            emotions_generation_prompt=emotions_generation_prompt,
            emotional_agent_prompt=emotional_agent_prompt,
            emotional_agent_debate_prompt=emotional_agent_debate_prompt,
            aggregator_prompt=aggregator_prompt
        )
        pipeline = Pipeline(inside_out_pipeline_config, llm_client)

        logger.info(f"Start inference on {len(dset)} dialogues")
        predictions, gt = run_pipeline(pipeline, inside_out_pipeline_config, dset, args.num_workers)
        with open(args.out_path, "w") as f:
            json.dump({"predictions": predictions}, f)
        logger.info(f"Saved results to {args.out_path}")

        logger.info(f"Accuracy: {accuracy(gt, [pred.split(";")[0].lower() for pred in predictions])}")


    logger.info(f"Completion tokens: {llm_client.get_output_tokens().sum()}")
    logger.info(f"Prompt tokens: {llm_client.get_input_tokens().sum()}")
    logger.info(f"Generation cost: {llm_client.get_generations_cost()}")

if __name__ == "__main__":
    main()