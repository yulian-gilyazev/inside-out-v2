import argparse
import copy
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from src.agent import Agent, AgentConfig, AgentContext, AgentFactory, Pipeline, PipelineAgentConfig
from src.llm_client import LLMClient
from src.schema.llm_config import LLMConfig
from src.scripts.experiments.agents_opro import OptimizePipelineConfig, optimize_pipeline
from src.scripts.experiments.utils import accuracy, run_pipeline
from src.utils.data import EmpatheticDialoguesDataset, SyntheticEmotionDataset, split_dataset
from src.utils.logger import Logger
from src.utils.prompts import get_emotions_generation_prompt, get_inside_out_aggregator_prompt, get_inside_out_emotinoal_prompt, get_system_prompt

"""
python3 -m src.scripts.experiments.inside-out-alt-topology --action 'evaluate' --dataset 'empatheticdialogues' \
      --dataset_path 'data/empatheticdialogues' --part 'test' \
      --out_path 'data/debug/empatheticdialogues_test_inside_out_erc_alt_topology_extended_emotions_results_gpt4o.json' --is_extended \
      --llm_config_path 'configs/llm_generation/openai_gpt_4o_config.json' \
      --train_size 200 --test_size 100 --num_workers 4
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
            current_res = self.io_agent.handle(curr_context)
            results.append(current_res)
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
    "ListConcatenator", 
    ListConcatenatorAgent,
    ListConcatenatorAgentConfig, 
    use_llm=False
)


def get_inside_out_exp_pipeline_cfg(
    system_prompt: str,
    emotions_generation_prompt: str, 
    emotional_agent_prompt: str,
    aggregator_prompt: str,
):
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
                    {"role": "user", "content": "Generate a set of emotional states for the following dialogue:\n\n{input}."},
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
                    {"role": "system", "content": system_prompt + "\n" + emotional_agent_prompt},
                    {"role": "user", "content": "Classify the emotion of speaker (A) in the following dialogue:\n\n{input}."}
                ]
            },
            {
                "agent_type": "ListConcatenator",
                "agent_id": "inside_out_concatenator",
                "input_id": "inside_out_agents",
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
                    {
                        "role": "user",
                        "content": "Classify the emotion of speaker (A) in the following dialogue using the following agent responses\n\nAgent responses:\n* {inside_out_concatenator}\n\n\nDialogue:\n{input}"
                    }
                ]
            },
        ],
        edges=[("input", "emotions_generator"), ("emotions_generator", "emotion_parser"), ("emotion_parser", "inside_out_agents"), ("inside_out_agents", "inside_out_concatenator"), ("inside_out_concatenator", "aggregator")],
        input_id="input",
        output_id="aggregator",
    )
    return pipeline_config


opro_metaprompt = """You are an AI assistant specializing in optimizing prompts for emotion classification.

## Task Description
Optimize an Emotion Recognition prompts for multi-agent pipeline for emotion classification tasks. The pipeline must accurately classify the emotion of the first (A) interlocutor in dialogues into one of the categories, along with a confidence score.

## Pipeline Specifications
- Powered by Large Language Models (LLMs)
- Leverages inter-agent communication between LLM calls for enhanced performance
- Implements inside-out idea, where the pipeline is composed of multiple agents, each of which is an LLM call
- First agent generates emotions
- Other agents implement inside-out idea, where they are prompted to feel the emotion generated by the first agent and then to assess the emotion of the first interlocutor in the dialogue.
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

opro_task_prompt = """Analyze previous prompts and create an optimized version that outperforms all prior examples in terms of quality and accuracy scores. Focus on refining agent interactions and prompt engineering to maximize emotion classification performance. Your response must contain ONLY the configuration JSON, enclosed within <PROMPT> and </PROMPT> tags. Do not include any explanations, comments, or additional text outside these tags."""

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=["synthetic", "empatheticdialogues"], help='Dataset to use')
    parser.add_argument('--dataset_path', type=str, help='Path to dataset')
    parser.add_argument('--part', type=str, choices=["train", "dev", "test"], required=False, help='Part of dataset to use')
    parser.add_argument('--is_extended', action='store_true', help='Use extended emotions set')
    parser.add_argument('--test_size', type=int, default=30, help='Number of dialogues to use for testing')
    parser.add_argument('--train_size', type=int, required=False, help='Number of dialogues to use for training')
    parser.add_argument('--llm_config_path', type=str,
                        default="configs/llm_generation/openai_gpt_4o_mini_config.json", help='Path to llm config')
    parser.add_argument('--llm_prompt_searcher_config_path', type=str,
                        default="configs/llm_generation/openai_gpt_4o_config.json", help='Path to llm config for prompt searcher')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers to use')
    parser.add_argument('--action', type=str, default="evaluate", choices=["evaluate", "optimize"], help='Action to perform')
    parser.add_argument('--out_path', type=str, required=False, help='Path where results will be saved')
    args = parser.parse_args()
    if args.dataset == "empatheticdialogues":
       assert args.part is not None, "Part must be specified for synthetic dataset"
    if args.dataset == "synthetic":
        assert args.is_extended is False, "Emotions set must be base for synthetic dataset"
    if args.action == "optimize":
        assert args.test_size is not None, "Test size must be specified for optimization"
        assert args.train_size is not None, "Train size must be specified for evaluation"
    if args.action == "evaluate":
        assert args.out_path is not None, "Output path must be specified for evaluation"
    return args

def main():
    args = parse_arguments()

    with open(args.llm_config_path, "r") as f:
        config_dct = json.load(f)
    llm_client = LLMClient(LLMConfig.from_dict(config_dct))

    system_prompt = get_system_prompt()

    emotions_generation_prompt = get_emotions_generation_prompt(is_extended=args.is_extended)

    emotional_agent_prompt = get_inside_out_emotinoal_prompt(is_extended=args.is_extended)

    emotional_agent_prompt = emotional_agent_prompt.replace("{emotion}", "{emotion_parser}")

    aggregator_prompt = get_inside_out_aggregator_prompt(is_extended=args.is_extended)

    if args.dataset == "synthetic":
        dialogues_path = os.path.join(args.dataset_path, "dialogues.json")
        scenarios_path = os.path.join(args.dataset_path, "scenarios.json")
        dset = SyntheticEmotionDataset(dialogues_path, scenarios_path)
        dset, train_dset = split_dataset(dset, args.test_size)
        train_dset, _ = split_dataset(train_dset, args.train_size)
    elif args.dataset == "empatheticdialogues":
        dset = EmpatheticDialoguesDataset(args.dataset_path, args.part, extended=args.is_extended)
        dset, _ = split_dataset(dset, args.test_size)
        train_dset = EmpatheticDialoguesDataset(args.dataset_path, "train", extended=args.is_extended)
        train_dset, _ = split_dataset(train_dset, args.train_size)

    if args.action == "optimize":
        with open(args.llm_prompt_searcher_config_path, "r") as f:
            config_dct = json.load(f)
        llm_prompt_searcher = LLMClient(LLMConfig.from_dict(config_dct))

        optimize_config = OptimizePipelineConfig(num_workers=args.num_workers)
        logger = Logger(
            group="inside-out-alt-topology-prompt-optimization",
            run_name="run_empatheticdialogues_v2_04_02",
            tags=["inside-out-alt-topology", "empatheticdialogues"],
            config=optimize_config.to_dict(),
            use_wandb=True,
        )

        prompts = {
            "system_prompt": system_prompt,
            "emotions_generation_prompt": emotions_generation_prompt,
            "emotional_agent_prompt": emotional_agent_prompt,
            "aggregator_prompt": aggregator_prompt,
        }

        optimize_pipeline(
            llm_client, 
            llm_prompt_searcher,
            logger,
            optimize_config,
            train_dset,
            dset,
            opro_metaprompt,
            opro_task_prompt,
            opro_prompt_tokens=("<PROMPT>", "</PROMPT>"),
            prompts=prompts, 
            cfg_from_prompts_fn=get_inside_out_exp_pipeline_cfg,
            check_fn=None)
    
    elif args.action == "evaluate":
        assert args.out_path is not None, "Output path must be specified"
        logger = Logger(
            use_wandb=False
        )
        inside_out_pipeline_config = get_inside_out_exp_pipeline_cfg(
            system_prompt=system_prompt,
            emotions_generation_prompt=emotions_generation_prompt,
            emotional_agent_prompt=emotional_agent_prompt,
            aggregator_prompt=aggregator_prompt
        )
        pipeline = Pipeline(inside_out_pipeline_config, llm_client)

        logger.info(f"Start inference on {len(dset)} dialogues")
        predictions, gt = run_pipeline(pipeline, inside_out_pipeline_config, dset, args.num_workers)
        with open(args.out_path, "w") as f:
            json.dump({"predictions": predictions}, f)
        logger.info(f"Saved results to {args.out_path}")


        logger.info(f"Accuracy: {accuracy(gt, [pred.split(';')[0].lower() for pred in predictions])}")
    
    logger.info(f"Completion tokens: {llm_client.get_output_tokens().sum()}")
    logger.info(f"Prompt tokens: {llm_client.get_input_tokens().sum()}")
    logger.info(f"Generation cost: {llm_client.get_generations_cost()}")

if __name__ == "__main__":
    main()