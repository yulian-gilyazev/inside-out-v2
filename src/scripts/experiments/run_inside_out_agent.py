import argparse
import json
import os
from functools import partial
from typing import List

from loguru import logger
from tqdm.auto import tqdm

from src.agent import AgentContext, Pipeline, registry
from src.agent.registry import PipelineAgentConfig
from src.llm_client import LLMClient
from src.schema.llm_config import LLMConfig
from src.scripts.experiments.agents_opro import (OptimizePipelineConfig,
                                                 optimize_pipeline)
from src.scripts.experiments.utils import (accuracy, evaluate_pipeline,
                                           run_pipeline)
from src.utils.data import (EmotionDataset, EmpatheticDialoguesDataset,
                            SyntheticEmotionDataset, split_dataset)
from src.utils.logger import Logger
from src.utils.prompts import (get_inside_out_aggregator_prompt,
                               get_inside_out_emotinoal_prompt,
                               get_system_prompt)

""" Example
python3 -m src.scripts.experiments.run_inside_out_agent --dataset 'synthetic' \
      --dataset_path 'data/synthetic_dialogues/v2' --out_path 'data/inside_out_erc_results.json'

python3 -m src.scripts.experiments.run_inside_out_agent  --dataset 'empatheticdialogues' \
      --dataset_path 'data/empatheticdialogues' --part 'test' --out_path 'data/empatheticdialogues_test_inside_out_erc_results.json'

python3 -m src.scripts.experiments.run_inside_out_agent --action 'evaluate' --dataset 'empatheticdialogues' \
      --dataset_path 'data/empatheticdialogues' --part 'test' \
      --out_path 'data/empatheticdialogues_test_inside_out_erc_extended_emotions_results_gpt4o.json' --is_extended \
      --llm_config_path 'configs/llm_generation/openai_gpt_4o_config.json' \
      --train_size 200 --test_size 1000 --num_workers 4
"""


def get_inside_out_exp_pipeline_cfg(system_prompt: str,
    emotional_agent_prompt: str,
    aggregator_prompt: str,
    emotions_list: List[str] = ("Anger", "Disgust", "Fear", "Happiness", "Sadness")):
        emotions_list_str = ", ".join(emotions_list)
        n_emotions = len(emotions_list)
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
                        "content": system_prompt + "\n" + emotional_agent_prompt.format(emotion="Anger", emotions_list=emotions_list_str, n_emotions=n_emotions)
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
                        "content": system_prompt + "\n" + emotional_agent_prompt.format(emotion="Disgust", emotions_list=emotions_list_str, n_emotions=n_emotions)
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
                        "content": system_prompt + "\n" + emotional_agent_prompt.format(emotion="Fear", emotions_list=emotions_list_str, n_emotions=n_emotions)
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
                        "content": system_prompt + "\n" + emotional_agent_prompt.format(emotion="Happiness", emotions_list=emotions_list_str, n_emotions=n_emotions)
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
                        "content": system_prompt + "\n" + emotional_agent_prompt.format(emotion="Sadness", emotions_list=emotions_list_str, n_emotions=n_emotions)
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

opro_metaprompt = """You are an AI assistant specializing in optimizing prompts for emotion classification.

## Task Description
Optimize an Emotion Recognition prompts for multi-agent pipeline for emotion classification tasks. The pipeline must accurately classify the emotion of the first (A) interlocutor in dialogues into one of the categories, along with a confidence score.

## Pipeline Specifications
- Powered by Large Language Models (LLMs)
- Leverages inter-agent communication between LLM calls for enhanced performance
- Agents implement inside-out idea, where they are prompted to feel the emotion generated by the first agent and then to assess the emotion of the first interlocutor in the dialogue.
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
    emotional_agent_prompt = get_inside_out_emotinoal_prompt(is_extended=args.is_extended)
    aggregator_prompt = get_inside_out_aggregator_prompt(is_extended=args.is_extended)
    if args.dataset == "synthetic":
        dialogues_path = os.path.join(args.dataset_path, "dialogues.json")
        scenarios_path = os.path.join(args.dataset_path, "scenarios.json")
        dset = SyntheticEmotionDataset(dialogues_path, scenarios_path)
        dset, train_dset = split_dataset(dset, 300)
        if args.action == "optimize":
            train_dset, _ = split_dataset(train_dset, args.train_size)
    elif args.dataset == "empatheticdialogues":
        dset = EmpatheticDialoguesDataset(args.dataset_path, args.part, extended=args.is_extended)
        dset, _ = split_dataset(dset, args.test_size)
        train_dset = EmpatheticDialoguesDataset(args.dataset_path, "train", extended=args.is_extended)
        if args.action == "optimize":
            train_dset, _ = split_dataset(train_dset, args.train_size)

    if args.action == "optimize":
        optimize_config = OptimizePipelineConfig(num_workers=args.num_workers)
        logger = Logger(
            group="inside-out-prompt-optimization",
            run_name="run_empatheticdialogues_v2_04_06",
            tags=["inside-out", "empatheticdialogues"],
            config=optimize_config.to_dict(),
            use_wandb=True,
        )
        with open(args.llm_prompt_searcher_config_path, "r") as f:
            config_dct = json.load(f)
        llm_prompt_searcher = LLMClient(LLMConfig.from_dict(config_dct))

        prompts = {
            "emotional_agent_prompt": emotional_agent_prompt,
            "aggregator_prompt": aggregator_prompt,
            "system_prompt": system_prompt,
        }
        emotions_list = ("Anger", "Disgust", "Fear", "Happiness", "Sadness")
        cfg_from_prompts_fn = partial(get_inside_out_exp_pipeline_cfg, emotions_list=emotions_list)

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
            cfg_from_prompts_fn=cfg_from_prompts_fn,
            check_fn=None
        )
    else:
        logger = Logger(
            use_wandb=False,
        )
        assert args.out_path is not None, "Output path must be specified"
        inside_out_pipeline_config = get_inside_out_exp_pipeline_cfg(
            system_prompt=system_prompt,
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