import argparse
import json
import os
from loguru import logger
from tqdm.auto import tqdm

from src.agent import Pipeline, AgentContext, registry
from src.llm_client import LLMClient
from src.schema.llm_config import LLMConfig
from src.utils.data import SyntheticEmotionDataset, EmpatheticDialoguesDataset, split_dataset


""" Example
python3 -m src.scripts.experiments.run_inside_out_agent --agent_name 'inside-out-erc' --dataset 'synthetic' \
      --dataset_path 'data/synthetic_dialogues/v2' --out_path 'data/inside_out_erc_results.json'

python3 -m src.scripts.experiments.run_inside_out_agent --agent_name 'inside-out-erc' --dataset 'empatheticdialogues' \
      --dataset_path 'data/empatheticdialogues' --part 'test' --out_path 'data/empatheticdialogues_test_inside_out_erc_results.json'

python3 -m src.scripts.experiments.run_inside_out_agent --agent_name 'inside-out-erc-extended-emotions' --dataset 'empatheticdialogues' \
      --dataset_path 'data/empatheticdialogues' --part 'test' --emotions_set 'extended' \
      --out_path 'data/empatheticdialogues_test_inside_out_erc_extended_emotions_results_gpt4o.json'
"""


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_name', type=str, help='Name of agent')
    parser.add_argument('--dataset', type=str, choices=["synthetic", "empatheticdialogues"], help='Dataset to use')
    parser.add_argument('--dataset_path', type=str, help='Path to dataset')
    parser.add_argument('--part', type=str, choices=["train", "dev", "test"], required=False, help='Part of dataset to use')
    parser.add_argument('--emotions_set', type=str, choices=["base", "extended"], default="base", help='Emotions set to use')
    parser.add_argument('--test_size', type=int, default=300, help='Number of dialogues to use for testing')
    parser.add_argument('--llm_config_path', type=str,
                        default="configs/llm_generation/openai_gpt_4o_config.json", help='Path to llm config')
    parser.add_argument('--out_path', type=str, help='Path where results will be saved')
    args = parser.parse_args()
    if args.dataset == "empatheticdialogues":
       assert args.part is not None, "Part must be specified for synthetic dataset"
       assert args.emotions_set is not None, "Emotions set must be specified for empatheticdialogues dataset"
    if args.dataset == "synthetic":
        assert args.emotions_set == "base", "Emotions set must be either base or extended for synthetic dataset"
    return args


def main():
    args = parse_arguments()

    with open(args.llm_config_path, "r") as f:
        config_dct = json.load(f)
    llm_client = LLMClient(LLMConfig.from_dict(config_dct))

    inside_out_pipeline_config = registry.get_config(args.agent_name)

    pipeline = Pipeline(inside_out_pipeline_config, llm_client)
    if args.dataset == "synthetic":
        dialogues_path = os.path.join(args.dataset_path, "dialogues.json")
        scenarios_path = os.path.join(args.dataset_path, "scenarios.json")
        dset = SyntheticEmotionDataset(dialogues_path, scenarios_path)
    elif args.dataset == "empatheticdialogues":
        dset = EmpatheticDialoguesDataset(args.dataset_path, args.part, extended=args.emotions_set == "extended")
    
    dset, _ = split_dataset(dset, args.test_size)
        
    logger.info(f"Start inference on {len(dset)} dialogues")
    result = []
    for item in tqdm(dset):

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