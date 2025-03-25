import json
import tempfile
import numpy as np
import os
from src.utils.logger import Logger
from src.models.opro import OPRO
from src.utils.data import SyntheticEmotionDataset, EmpatheticDialoguesDataset, split_dataset
from src.schema.llm_config import LLMConfig
from src.llm_client import LLMClient
from src.agent.pipeline import PipelineAgentConfig, Pipeline, AgentContext
from dataclasses import dataclass, asdict
from typing import Dict, Any, Literal
from tqdm import tqdm

"""Run
python3 -m src.scripts.experiments.optimize-inside-out-prompts
"""

def check_config(config: str) -> bool:
    """Simple check if the config is valid"""
    try:
        config = json.loads(config)
    except json.JSONDecodeError:
        return False

    required_keys = ["agent_configs", "edges", "input_id", "output_id"]
    if not all(key in config for key in required_keys):
        return False
        
    all_agent_ids = set()
    for agent_config in config["agent_configs"]:
        if not all(key in agent_config for key in ["agent_type", "agent_id"]):
            return False
        all_agent_ids.add(agent_config["agent_id"])
        
    if len(all_agent_ids) != len(config["agent_configs"]):
        return False
        
    return all(edge[0] in all_agent_ids and edge[1] in all_agent_ids 
               for edge in config["edges"])

opro_metaprompt = """You are an AI assistant specializing in optimizing multi-agent pipelines for emotion classification.

## Task Description
Optimize an Emotion Recognition multi-agent pipeline for emotion classification tasks. The pipeline must accurately classify the emotion of the first (A) interlocutor in dialogues into one of the following categories, along with a confidence score:
- Anger
- Disgust
- Fear
- Happiness
- Sadness

## Pipeline Specifications
- Powered by Large Language Models (LLMs)
- Leverages inter-agent communication between LLM calls for enhanced performance

## Input
- Historical pipeline configuration examples in JSON format with accuracy metrics (scale: 0 to 1), in format:
```
<CONFIG>{config}</CONFIG> 
accuracy: {accuracy}
```

## Modification Scope
You may modify:
- `messages` fields within `agent_configs`
- IMPORTANT: Do not alter template variables (e.g., {input}, {anger_agent}, {aggregator}) as they are essential for passing information between agents
- IMPORTANT: Do not change the number of agents, and idea of agent system. You can only change prompts, but idea of agents should be the same.

## Requirements
You must preserve:
- The fundamental pipeline architecture
- Core functionality and processing flow
- Key architectural components and their relationships

## Critical Guidelines
- Maintain the working structure of the pipeline
- Prioritize accuracy metric improvements
- Analyze previous high-performing examples for insights
- Ensure valid and properly formatted JSON output

Return your optimized pipeline configuration within <CONFIG> tags in valid JSON format, designed to maximize classification accuracy.
"""

task_prompt = """Analyze previous configurations and create an optimized version that outperforms all prior examples in terms of quality and accuracy scores. Focus on refining agent interactions and prompt engineering to maximize emotion classification performance. Your response must contain ONLY the configuration JSON, enclosed within <CONFIG> and </CONFIG> tags. Do not include any explanations, comments, or additional text outside these tags."""


def accuracy(gt, pred):
    mask = [u == v for u, v in zip(gt, pred)]
    return np.array(mask).mean()


@dataclass
class ExperimentConfig:
    llm_config_path: str = "configs/llm_generation/openai_gpt_4o_mini_config.json"
    llm_prompt_searcher_config_path: str = "configs/llm_generation/openai_gpt_4o_config.json"
    dataset: Literal["empatheticdialogues", "synthetic"] = "empatheticdialogues"
    dataset_path: str = "data/empatheticdialogues"
    emotions_set: Literal["base", "extended"] = "extended"
    initial_config_path: str = "src/scripts/experiments/inside-out-v1-base-config.json"
    n_iters: int = 15
    test_size: int = 200
    train_size: int = 200
    opro_memory_strategy: Literal["last", "all", "ascending_subsequence"] = "all"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    

def evaluate_pipeline(pipeline: Pipeline, pipeline_cfg: PipelineAgentConfig, dset: SyntheticEmotionDataset) -> float:
    predictions = []
    gt = []
    for item in tqdm(dset):
        context = AgentContext(data={"input": item.format_dialogue()})
        context = pipeline.process(context)
        predicted = context.get_value(pipeline_cfg.output_id)
        predicted = predicted.split(";")[0].lower()
        predictions.append(predicted)
        gt.append(item.emotion.value.lower())
    return accuracy(gt, predictions)


def main(config: ExperimentConfig):

    with open(config.llm_config_path, "r") as f:
        config_dct = json.load(f)
    llm_client = LLMClient(LLMConfig.from_dict(config_dct))

    with open(config.llm_prompt_searcher_config_path, "r") as f:
        config_dct = json.load(f)
    llm_prompt_searcher = LLMClient(LLMConfig.from_dict(config_dct))

    with open(config.initial_config_path, "r") as f:
        pipeline_cofig_str = f.read()


    logger = Logger(
        group="inside-out-v1-prompt-optimization",
        run_name="run_empatheticdialogues_extended_all",
        tags=["inside-out-v1", "erc"],
        config=config.to_dict(),
        use_wandb=True,
    )

    if config.dataset == "synthetic":
        dset = SyntheticEmotionDataset(os.path.join(config.dataset_path, "dialogues.json"), os.path.join(config.dataset_path, "scenarios.json"))
        dset_test, dset = split_dataset(dset, config.test_size)
        dset_test.shuffle()
        dset_train, _ = split_dataset(dset, config.train_size)
    elif config.dataset == "empatheticdialogues":
        dset_test = EmpatheticDialoguesDataset(config.dataset_path, part="test", extended=config.emotions_set == "extended")
        dset_test, _ = split_dataset(dset_test, config.test_size)
        dset_test.shuffle()
        dset_train = EmpatheticDialoguesDataset(config.dataset_path, part="train", extended=config.emotions_set == "extended")
        dset_train, _ = split_dataset(dset_train, config.train_size)

    logger.info(f"Start OPRO optimization")
    optimizer = OPRO(llm_prompt_searcher, "accuracy", opro_metaprompt, task_prompt, check_fn=check_config, prompt_tokens=("<CONFIG>", "</CONFIG>"), memory_strategy=config.opro_memory_strategy, logger=logger)

    for step in tqdm(range(config.n_iters)):
        pipeline_cfg = PipelineAgentConfig(**json.loads(pipeline_cofig_str))
        pipeline = Pipeline(pipeline_cfg, llm_client)
        train_acc = evaluate_pipeline(pipeline, pipeline_cfg, dset_train)
        logger.log(
            metric_name="train_accuracy",
            value=train_acc,
            log_stdout=True,
            log_wandb=True,
        )
        test_acc = evaluate_pipeline(pipeline, pipeline_cfg, dset_test)
        logger.log(
            metric_name="test_accuracy",
            value=test_acc,
            log_stdout=True,
            log_wandb=True,
        )
        if step == 0:
            optimizer.initialize(pipeline_cofig_str, reward=train_acc)
            new_prompt = optimizer.step()
            pipeline_cofig_str = new_prompt
        else:
            optimizer.send_reward(train_acc)
            new_prompt = optimizer.step()
            pipeline_cofig_str = new_prompt
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            cfg_path = f.name
            f.write(pipeline_cofig_str.encode("utf-8"))
            artifact = logger.wandb.Artifact(name=f"config_{step}", type="dataset")
            artifact.add_file(cfg_path)
            logger.run.log_artifact(artifact)
            logger.info(f"Config {step} saved")
            logger.info(f"Config: \n{pipeline_cofig_str}")
    
    logger.info(f"Completion tokens: {llm_client.get_output_tokens().sum()}")
    logger.info(f"Prompt tokens: {llm_client.get_input_tokens().sum()}")
    logger.info(f"Generation cost: {llm_client.get_generations_cost()}")


if __name__ == "__main__":
    config = ExperimentConfig(
        llm_config_path="configs/llm_generation/openai_gpt_4o_mini_config.json",
        llm_prompt_searcher_config_path="configs/llm_generation/openai_gpt_4o_config.json",
        dataset="empatheticdialogues",
        dataset_path="data/empatheticdialogues",
        emotions_set="extended",
        initial_config_path="src/scripts/experiments/inside-out-v1-base-extended-emotions-config.json",
        n_iters=15,
        test_size=300,
        train_size=300,
        opro_memory_strategy="all",
    )
    main(config)