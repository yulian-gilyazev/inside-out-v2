import argparse
import json
import os
from functools import partial
from typing import List, Callable, Optional, Tuple

from loguru import logger
from tqdm.auto import tqdm

from src.agent import AgentContext, Pipeline, registry
from src.agent.registry import PipelineAgentConfig
from src.llm_client import LLMClient
from src.schema.emotions import EmotionSet
from src.schema.llm_config import LLMConfig
from src.scripts.experiments.agents_opro import OptimizePipelineConfig, optimize_pipeline
from src.scripts.experiments.utils import (accuracy, evaluate_pipeline,
                                           run_pipeline)
from src.utils.data import (EmotionDataset, EmpatheticDialoguesDataset, Dialogue,
                            SyntheticEmotionDataset, split_dataset)
from src.utils.logger import Logger
from src.utils.prompts import (get_baseline_zero_shot_prompt, get_system_prompt)
import tempfile
from src.models.opro import OPRO

""" Example
python3 -m src.scripts.experiments.baseline_llm --action 'evaluate' --dataset 'empatheticdialogues' \
      --dataset_path 'data/empatheticdialogues' --part 'test' \
      --out_path 'data/debug/empatheticdialogues_test_baseline_truncated_gpt4o.json' --emotions_set 'truncated' \
      --llm_config_path 'configs/llm_generation/openai_gpt_4o_config.json' \
      --train_size 200 --test_size 1000 --num_workers 16
"""


def get_baseline_pipeline_cfg(system_prompt: str, baseline_prompt: str):
    """
    Создает конфигурацию бейзлайн пайплайна.
    """
    pipeline_config = PipelineAgentConfig(
        agent_configs=[
            {
                "agent_type": "Echo",
                "agent_id": "input",
            },
            {
                "agent_type": "IO",
                "agent_id": "classifier",
                "messages": [
                    {"role": "system", "content": system_prompt + baseline_prompt},
                    {"role": "user", "content": "Classify the following dialogue:\n\n{input}."},
                ]
            },
        ],
        edges=[("input", "classifier")],
        input_id="input",
        output_id="classifier",
    )
    return pipeline_config


opro_metaprompt = """You are an AI assistant specializing in optimizing a single prompt for emotion classification.

## Task Description
Optimize a single Emotion Recognition prompt for a Large Language Model (LLM) to classify the emotion of the first (A) interlocutor in dialogues into one of the categories, along with a confidence score.

## Prompt Specifications
- Powered by a Large Language Model (LLM)
- The prompt should guide the LLM to accurately assess the emotion of the first interlocutor in the dialogue.
- The prompt should be clear and concise to maximize the LLM's performance.

## Input
- Historical prompt examples in JSON format with accuracy metrics (scale: 0 to 1), in format:
```
<PROMPT>{prompt}</PROMPT> 
accuracy: {accuracy}
```

Return your optimized prompt within <PROMPT> tags, designed to maximize classification accuracy. Before returning the prompt, you are allowed to give analysis of the previous prompts and the results and suggest changes. But stick to the format given in the description.
"""

opro_task_prompt = """Analyze previous prompts and create an optimized version that outperforms all prior examples in terms of quality and accuracy scores. Your response must contain ONLY the prompt, enclosed within <PROMPT> and </PROMPT> tags. Do not include any explanations, comments, or additional text outside these tags."""


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=["synthetic", "empatheticdialogues"], help='Датасет для использования')
    parser.add_argument('--dataset_path', type=str, help='Путь к датасету')
    parser.add_argument('--part', type=str, choices=["train", "dev", "test"], required=False, help='Часть датасета для использования')
    parser.add_argument('--emotions_set', type=str, choices=["base", "emp_dialogues", "truncated"], help='Набор эмоций для использования')
    parser.add_argument('--test_size', type=int, default=30, help='Количество диалогов для тестирования')
    parser.add_argument('--train_size', type=int, required=False, help='Количество диалогов для обучения')
    parser.add_argument('--llm_config_path', type=str,
                        default="configs/llm_generation/openai_gpt_4o_mini_config.json", help='Путь к конфигурации LLM')
    parser.add_argument('--llm_prompt_searcher_config_path', type=str,
                        default="configs/llm_generation/openai_gpt_4o_config.json", help='Путь к конфигурации LLM для поиска промптов')
    parser.add_argument('--num_workers', type=int, default=4, help='Количество рабочих процессов')
    parser.add_argument('--action', type=str, default="evaluate", choices=["evaluate", "optimize"], help='Действие для выполнения')
    parser.add_argument('--out_path', type=str, required=False, help='Путь для сохранения результатов')
    args = parser.parse_args()
    if args.dataset == "empatheticdialogues":
       assert args.part is not None, "Необходимо указать часть для датасета EmpatheticDialogues"
    if args.dataset == "synthetic":
        assert args.is_extended is False, "Набор эмоций должен быть базовым для синтетического датасета"
    if args.action == "optimize":
        assert args.test_size is not None, "Необходимо указать размер тестовой выборки для оптимизации"
        assert args.train_size is not None, "Необходимо указать размер обучающей выборки для оптимизации"
    if args.action == "evaluate":
        assert args.out_path is not None, "Необходимо указать путь для сохранения результатов"
    args.emotion_set = EmotionSet.from_str(args.emotions_set)
    return args


def main():
    args = parse_arguments()

    with open(args.llm_config_path, "r") as f:
        config_dct = json.load(f)
    llm_client = LLMClient(LLMConfig.from_dict(config_dct))

    system_prompt = get_system_prompt()
    baseline_prompt = get_baseline_zero_shot_prompt(args.emotion_set)

    print(baseline_prompt)
    print(system_prompt)
    if args.dataset == "synthetic":
        dialogues_path = os.path.join(args.dataset_path, "dialogues.json")
        scenarios_path = os.path.join(args.dataset_path, "scenarios.json")
        dset = SyntheticEmotionDataset(dialogues_path, scenarios_path)
        dset, train_dset = split_dataset(dset, 300)
        if args.action == "optimize":
            train_dset, _ = split_dataset(train_dset, args.train_size)
    elif args.dataset == "empatheticdialogues":
        dset = EmpatheticDialoguesDataset(args.dataset_path, args.part, emotion_set=args.emotion_set)
        dset, _ = split_dataset(dset, args.test_size)
        train_dset = EmpatheticDialoguesDataset(args.dataset_path, "train", emotion_set=args.emotion_set)
        if args.action == "optimize":
            train_dset, _ = split_dataset(train_dset, args.train_size)

    if args.action == "optimize":
        optimize_config = OptimizePipelineConfig(num_workers=args.num_workers)
        logger = Logger(
            group="inside-out-prompt-optimization",
            run_name="run_baseline_optimization_16_04",
            tags=["baseline", args.dataset],
            config=optimize_config.to_dict(),
            use_wandb=True,
        )
        with open(args.llm_prompt_searcher_config_path, "r") as f:
            config_dct = json.load(f)
        llm_prompt_searcher = LLMClient(LLMConfig.from_dict(config_dct))

        prompts = baseline_prompt
        cfg_from_prompts_fn = lambda prompt: get_baseline_pipeline_cfg(system_prompt, prompt.strip('"'))
        
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
        assert args.out_path is not None, "You must specify the path to save the results"
        baseline_pipeline_config = get_baseline_pipeline_cfg(
            system_prompt=system_prompt,
            baseline_prompt=baseline_prompt
        )
        pipeline = Pipeline(baseline_pipeline_config, llm_client)

        logger.info(f"Starting inference on {len(dset)} dialogues")
        predictions, gt = run_pipeline(pipeline, baseline_pipeline_config, dset, args.num_workers)
        with open(args.out_path, "w") as f:
            json.dump({"predictions": predictions}, f)
        logger.info(f"Results saved in {args.out_path}")
        logger.info(f"Accuracy: {accuracy(gt, [pred.split(';')[0].lower() for pred in predictions])}")

    logger.info(f"Output tokens: {llm_client.get_output_tokens().sum()}")
    logger.info(f"Input tokens: {llm_client.get_input_tokens().sum()}")
    logger.info(f"Generation cost: {llm_client.get_generations_cost()}")


if __name__ == "__main__":
    main()