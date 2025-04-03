import argparse
from dataclasses import dataclass, asdict
import json
import os
import numpy as np
from typing import List, Dict, Tuple, Any, Literal
import re
import copy
from src.agent import PipelineAgentConfig, AgentConfig, IOAgentConfig, Agent, AgentContext, AgentFactory, Pipeline
from src.agent import IOAgent
from src.llm_client import LLMClient
from src.schema.llm_config import LLMConfig
from src.utils.data import EmotionDataset, SyntheticEmotionDataset, EmpatheticDialoguesDataset, split_dataset
from src.utils.logger import Logger
from src.utils.prompts import get_inside_out_emotinoal_prompt, get_inside_out_aggregator_prompt, get_system_prompt, get_emotions_generation_prompt
from src.models.opro import OPRO
from tqdm.auto import tqdm
from loguru import logger
import tempfile


"""
python3 -m src.scripts.experiments.inside-out-alt-topology --dataset 'synthetic' --dataset_path 'data/synthetic_dialogues/v2' --out_path 'data/debug/inside_out_alt_topology_exp_04_02_25.json'

python3 -m src.scripts.experiments.inside-out-alt-topology --dataset 'empatheticdialogues' --dataset_path 'data/empatheticdialogues' --part 'test' --out_path 'data/debug/empatheticdialogues_test_inside_out_alt_topology_exp_04_02_25.json' --is_extended
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
                    {"role": "system", "content": system_prompt + "\n" + emotional_agent_prompt},
                     {"role": "user", "content": "Dialogue:\n{input}."}
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
                        "content": "Dialogue:\n{input}\nAgent responses:\n* {inside_out_concatenator}"
                    }
                ]
            },
        ],
        edges=[("input", "emotions_generator"), ("emotions_generator", "emotion_parser"), ("emotion_parser", "inside_out_agents"), ("inside_out_agents", "inside_out_concatenator"), ("inside_out_concatenator", "aggregator")],
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
                        default="configs/llm_generation/openai_gpt_4o_mini_config.json", help='Path to llm config')
    parser.add_argument('--out_path', type=str, help='Path where scenarios will be saved')
    parser.add_argument('--is_extended', action='store_true', help='Use extended dataset')
    args = parser.parse_args()
    if args.dataset == "empatheticdialogues":
        assert args.part is not None, "Part must be specified for synthetic dataset"
    return args


def accuracy(gt, pred):
    mask = [u == v for u, v in zip(gt, pred)]
    return np.array(mask).mean()

def evaluate_pipeline(pipeline: Pipeline, pipeline_cfg: PipelineAgentConfig, dset: SyntheticEmotionDataset) -> float:
    predictions = []
    gt = []
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial
    
    def process_item(item, pipeline, pipeline_cfg):
        context = AgentContext(data={"input": item.format_dialogue()})
        context = pipeline.process(context)
        predicted = context.get_value(pipeline_cfg.output_id)
        predicted = predicted.split(";")[0].lower()
        return predicted, item.emotion.value.lower()
    
    process_func = partial(process_item, pipeline=pipeline, pipeline_cfg=pipeline_cfg)
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(process_func, dset), total=len(dset)))
    
    predictions = [res[0] for res in results]
    gt = [res[1] for res in results]
    
    return accuracy(gt, predictions)


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

task_prompt = """Analyze previous prompts and create an optimized version that outperforms all prior examples in terms of quality and accuracy scores. Focus on refining agent interactions and prompt engineering to maximize emotion classification performance. Your response must contain ONLY the configuration JSON, enclosed within <PROMPT> and </PROMPT> tags. Do not include any explanations, comments, or additional text outside these tags."""


@dataclass
class OptimizePipelineConfig:
    n_steps: int = 4
    llm_config_prompt_search_path: str = "configs/llm_generation/openai_gpt_4o_config.json"
    opro_memory_strategy: Literal["last", "all"] = "all"


    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def optimize_pipeline(args, 
                      optimize_config: OptimizePipelineConfig,
                      train_dset: EmotionDataset,
                      test_dset: EmotionDataset,
                      system_prompt: str,
                      emotions_generation_prompt: str,
                      emotional_agent_prompt: str,
                      aggregator_prompt: str):
    
    logger = Logger(
        group="inside-out-alt-topology-prompt-optimization",
        run_name="run_empatheticdialogues_v2_04_02",
        tags=["inside-out-alt-topology", "empatheticdialogues"],
        config=optimize_config.to_dict(),
        use_wandb=True,
    )
     
    with open(args.llm_config_path, "r") as f:
        config_dct = json.load(f)
    llm_client = LLMClient(LLMConfig.from_dict(config_dct))

    with open(optimize_config.llm_config_prompt_search_path, "r") as f:
        config_dct = json.load(f)
    llm_prompt_searcher = LLMClient(LLMConfig.from_dict(config_dct))


    optimizer = OPRO(llm_prompt_searcher,
                    "accuracy", 
                    opro_metaprompt,
                    task_prompt, 
                    check_fn=lambda x: True, 
                    prompt_tokens=("<PROMPT>", "</PROMPT>"),
                    memory_strategy=optimize_config.opro_memory_strategy,
                    logger=logger)
    
    pipeline_prompts_json = json.dumps({
        "system_prompt": system_prompt,
        "emotions_generation_prompt": emotions_generation_prompt,
        "emotional_agent_prompt": emotional_agent_prompt,
        "aggregator_prompt": aggregator_prompt,
    })

    for step in tqdm(range(optimize_config.n_steps)):
        pipeline_prompts = json.loads(pipeline_prompts_json)
        pipeline_prompts["emotional_agent_prompt"] = pipeline_prompts["emotional_agent_prompt"].replace("{emotion}", "{emotion_parser}")

        pipeline_cfg = get_inside_out_exp_pipeline_cfg(**pipeline_prompts)
        pipeline = Pipeline(pipeline_cfg, llm_client)
        train_acc = evaluate_pipeline(pipeline, pipeline_cfg, train_dset)
        logger.log(
            metric_name="train_accuracy",
            value=train_acc,
            log_stdout=True,
            log_wandb=True,
        )
        test_acc = evaluate_pipeline(pipeline, pipeline_cfg, test_dset)
        logger.log(
            metric_name="test_accuracy",
            value=test_acc,
            log_stdout=True,
            log_wandb=True,
        )
        if step == 0:
            optimizer.initialize(pipeline_prompts_json, reward=train_acc)
            new_prompt = optimizer.step()
            pipeline_prompts_json = new_prompt
        else:
            optimizer.send_reward(train_acc)
            new_prompt = optimizer.step()
            pipeline_prompts_json = new_prompt
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            cfg_path = f.name
            f.write(pipeline_prompts_json.encode("utf-8"))
            artifact = logger.wandb.Artifact(name=f"config_{step}", type="dataset")
            artifact.add_file(cfg_path)
            logger.run.log_artifact(artifact)
            logger.info(f"Config {step} saved")
            logger.info(f"Config: \n{pipeline_prompts_json}")
    
    logger.info(f"Completion tokens: {llm_client.get_output_tokens().sum()}")
    logger.info(f"Prompt tokens: {llm_client.get_input_tokens().sum()}")
    logger.info(f"Generation cost: {llm_client.get_generations_cost()}")
    return pipeline

def main():
    args = parse_arguments()

    with open(args.llm_config_path, "r") as f:
        config_dct = json.load(f)
    llm_client = LLMClient(LLMConfig.from_dict(config_dct))

    system_prompt = get_system_prompt()

    emotions_generation_prompt = get_emotions_generation_prompt(is_extended=args.is_extended)

    emotional_agent_prompt = get_inside_out_emotinoal_prompt(is_extended=args.is_extended)

    aggregator_prompt = get_inside_out_aggregator_prompt(is_extended=args.is_extended)

    print(emotions_generation_prompt)
    print(emotional_agent_prompt)
    print(aggregator_prompt)


    # inside_out_pipeline_config = get_inside_out_exp_pipeline_cfg(
    #     system_prompt=system_prompt,
    #     emotions_generation_prompt=emotions_generation_prompt,
    #     emotional_agent_prompt=emotional_agent_prompt,
    #     aggregator_prompt=aggregator_prompt
    # )

    # pipeline = Pipeline(inside_out_pipeline_config, llm_client)
    if args.dataset == "synthetic":
        dialogues_path = os.path.join(args.dataset_path, "dialogues.json")
        scenarios_path = os.path.join(args.dataset_path, "scenarios.json")
        dset = SyntheticEmotionDataset(dialogues_path, scenarios_path)
        dset, train_dset = split_dataset(dset, 300)
        train_dset, _ = split_dataset(train_dset, 300)
    elif args.dataset == "empatheticdialogues":
        dset = EmpatheticDialoguesDataset(args.dataset_path, args.part, extended=args.is_extended)
        dset, _ = split_dataset(dset, 300)
        train_dset = EmpatheticDialoguesDataset(args.dataset_path, "train", extended=args.is_extended)
        train_dset, _ = split_dataset(train_dset, 300)

    # if args.optimize_pipeline:

    optimize_pipeline(args,
                    optimize_config=OptimizePipelineConfig(n_steps=10),
                    train_dset=train_dset,
                    test_dset=dset,
                    system_prompt=system_prompt,
                    emotions_generation_prompt=emotions_generation_prompt,
                    emotional_agent_prompt=emotional_agent_prompt,
                    aggregator_prompt=aggregator_prompt)
        
    # else:

    #     logger.info(f"Start inference on {len(dset)} dialogues")
    #     result = []s
    #     for idx in tqdm(range(len(dset))):
    #         item = dset[idx]

    #         context = AgentContext(data={"input": item.format_dialogue()})

    #         context = pipeline.process(context)
    #         predicted = context.get_value(inside_out_pipeline_config.output_id)
    #         result.append({"id": item.id, "empathy_label": item.empathy_label, "prediction": predicted})

    #     with open(args.out_path, "w") as f:
    #         json.dump({"predictions": result}, f)
    #     logger.info(f"Saved results to {args.out_path}")

    logger.info(f"Completion tokens: {llm_client.get_output_tokens().sum()}")
    logger.info(f"Prompt tokens: {llm_client.get_input_tokens().sum()}")
    logger.info(f"Generation cost: {llm_client.get_generations_cost()}")


if __name__ == "__main__":
    main()