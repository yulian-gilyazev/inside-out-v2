import argparse
import json
from loguru import logger
from tqdm.auto import tqdm

from src.agent import Pipeline, AgentContext, registry
from src.llm_client import LLMClient
from src.schema.llm_config import LLMConfig
from src.utils.data import SyntheticEmotionDataset

""" Example
python3 -m src.scripts.experiments.run_inside_out_agent --agent_name 'inside-out-erc' \
    --out_path 'data/inside_out_erc_results.json'
"""


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_name', type=str, help='Name of agent')
    parser.add_argument('--dialogues_path', type=str,
                        default="data/synthetic_dialogues/v2/dialogues.json", help='Path to dialogues ')
    parser.add_argument('--scenarios_path', type=str,
                        default="data/synthetic_dialogues/v2/scenarios.json", help='Path to scenarios')
    parser.add_argument('--llm_config_path', type=str,
                        default="configs/openai_gpt_4o_config.json", help='Path to llm config')
    parser.add_argument('--out_path', type=str, help='Path where scenarios will be saved')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    with open(args.llm_config_path, "r") as f:
        config_dct = json.load(f)
    llm_client = LLMClient(LLMConfig.from_dict(config_dct))

    inside_out_pipeline_config = registry.get_config(args.agent_name)

    pipeline = Pipeline(inside_out_pipeline_config, llm_client)
    dset = SyntheticEmotionDataset(args.dialogues_path, args.scenarios_path)

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
    logger.info(f"Generation cost: {llm_client.get_generations_cost().shape[0]}")


if __name__ == "__main__":
    main()