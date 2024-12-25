import json
from loguru import logger

from src.agent import Pipeline, PipelineAgentConfig, IOAgent, AgentContext, registry
from src.llm_client import LLMClient
from src.schema.llm_config import LLMConfig
from src.utils.data import SyntheticEmotionDataset


def main():
    with open("configs/gpt_4o_mini_config.json", "r") as f:
        config_dct = json.load(f)
    llm_config = LLMConfig.from_dict(config_dct)
    llm_client = LLMClient(llm_config)

    inside_out_pipeline_config = registry.get_config("self-consistency-erc")

    pipeline = Pipeline(inside_out_pipeline_config, llm_client)
    dset = SyntheticEmotionDataset(["data/synthetic_dialogues/empathetic_dialogues.json",
                                    "data/synthetic_dialogues/non_empathetic_dialogues.json"],
                                   [1, 0], "data/synthetic_dialogues/scenarios.json")

    first_item, _ = dset[15]
    context = AgentContext(data={"input": first_item.format_dialogue()})

    context = pipeline.process(context)

    print("predicted: ", context.get_value(inside_out_pipeline_config.output_id))
    print(f"gt emotion: {first_item.emotion}")

    print("anger_agent: ", context.get_value("anger_agent"))
    print("disgust_agent: ", context.get_value("disgust_agent"))
    print("fear_agent: ", context.get_value("fear_agent"))
    print("happiness_agent: ", context.get_value("happiness_agent"))
    print("sadness_agent: ", context.get_value("sadness_agent"))

    logger.info(f"Completion tokens: {llm_client.get_output_tokens().sum()}")
    logger.info(f"Prompt tokens: {llm_client.get_input_tokens().sum()}")
    logger.info(f"LLM queries num: {llm_client.get_output_tokens().shape[0]}")


if __name__ == "__main__":
    main()