import argparse
import json
from loguru import logger
from tqdm.auto import tqdm

from src.agent import Pipeline, PipelineAgentConfig, IOAgent, AgentContext, registry
from src.llm_client import LLMClient
from src.schema.llm_config import LLMConfig
from src.utils.data import SyntheticEmotionDataset

""" Example
python3 -m src.scripts.experiments.run_inside_out_agent --agent_name 'baseline-erc' \
    --out_path 'data/baseline_erc_results.json'
"""


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_name', type=str, help='Name of agent')
    parser.add_argument('--empathetic_dialogues_path', type=str,
                        default="data/synthetic_dialogues/empathetic_dialogues.json", help='Path to llm config')
    parser.add_argument('--non_empathetic_dialogues_path', type=str,
                        default="data/synthetic_dialogues/non_empathetic_dialogues.json", help='Path to llm config')
    parser.add_argument('--scenarios_path', type=str,
                        default="data/synthetic_dialogues/scenarios.json", help='Path to llm config')
    parser.add_argument('--llm_config_path', type=str,
                        default="configs/gpt_4o_mini_config.json", help='Path to llm config')
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
    dset = SyntheticEmotionDataset([args.empathetic_dialogues_path,
                                    args.non_empathetic_dialogues_path],
                                   [1, 0], args.scenarios_path)
    result = []
    for idx in tqdm(range(200)):
        item, empathy_label = dset[idx]

        context = AgentContext(data={"input": item.format_dialogue()})

        context = pipeline.process(context)
        predicted = context.get_value(inside_out_pipeline_config.output_id)
        result.append({"idx": idx, "empathy_label": empathy_label, "predicted": predicted})
    with open(args.out_path, "w") as f:
        json.dump({"predictions": result}, f)
        
    logger.info(f"Completion tokens: {llm_client.get_output_tokens().sum()}")
    logger.info(f"Prompt tokens: {llm_client.get_input_tokens().sum()}")
    logger.info(f"LLM queries num: {llm_client.get_output_tokens().shape[0]}")


if __name__ == "__main__":
    main()