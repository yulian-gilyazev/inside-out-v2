import json
import tempfile
import numpy as np
from src.utils.logger import Logger
from src.models.opro import OPRO
from src.utils.data import SyntheticEmotionDataset, split_dataset
from src.schema.llm_config import LLMConfig
from src.llm_client import LLMClient
from src.agent.pipeline import PipelineAgentConfig, Pipeline, AgentContext
from dataclasses import dataclass, asdict
from typing import Dict, Any
from tqdm import tqdm

"""Запуск
python3 -m src.scripts.experiments.optimize-inside-out-prompts
"""

def check_config(config: str) -> bool:
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
Optimize a Emotion Recognition multi-agent pipeline for the emotion classification task. Pipeline should be able to classify the emotion of the first (A) interlocutor in the dialogue into one of the following emotions, and give a confidence score for the prediction:
- Anger
- Disgust
- Fear
- Happiness
- Sadness

## Pipeline Specifications
- Based on LLM
- Utilizes agent communication between LLM calls

## Input
- Pipeline configuration in json format
- Previous prompt examples with accuracy evaluations (0 to 1 scale)

## Modification Scope
You may modify:
- `messages` in `agent_configs`
- Be careful with template variables, they are not allowed to be changed, because they are used in the pipeline to pass the dialogue and other information to the agents

You must maintain:
- Core pipeline structure
- Basic functionality
- Main architectural components

## Important Notes
- Preserve the pipeline's working structure
- Focus on metric improvements
- Consider previous high-scoring examples
- Maintain proper JSON formatting

Return optimized pipeline configuration <CONFIG> in JSON format with improved classification metric.
"""

task_prompt = """Generate a configuration that exceeds the quality and score of all previous configs. Return only the configuration, nothing else, and separate it from the rest of the text with <CONFIG>...</CONFIG>."""


def accuracy(gt, pred):
    mask = [u == v for u, v in zip(gt, pred)]
    return np.array(mask).mean()


@dataclass
class ExperimentConfig:
    llm_config_path: str = "configs/llm_generation/openai_gpt_4o_mini_config.json"
    llm_prompt_searcher_config_path: str = "configs/llm_generation/openai_gpt_4o_config.json"
    dialogues_path: str = "data/synthetic_dialogues/v2/dialogues.json"
    scenarios_path: str = "data/synthetic_dialogues/v2/scenarios.json"
    initial_config_path: str = "src/scripts/experiments/inside-out-v1-base-config.json"
    n_iters: int = 14

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
        run_name="run_1",
        tags=["inside-out-v1", "erc"],
        config=config.to_dict(),
        use_wandb=True,
    )

    dset = SyntheticEmotionDataset(config.dialogues_path, config.scenarios_path)
    dset_test, dset = split_dataset(dset, 200)
    dset_train, _ = split_dataset(dset, 200)


    logger.info(f"Start OPRO optimization")
    optimizer = OPRO(llm_prompt_searcher, "accuracy", opro_metaprompt, task_prompt, check_fn=check_config, prompt_tokens=("<CONFIG>", "</CONFIG>"), logger=logger)

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
    

    logger.info(f"Completion tokens: {llm_client.get_output_tokens().sum()}")
    logger.info(f"Prompt tokens: {llm_client.get_input_tokens().sum()}")
    logger.info(f"Generation cost: {llm_client.get_generations_cost()}")


if __name__ == "__main__":
    main(ExperimentConfig())