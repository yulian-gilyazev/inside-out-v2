import json
from loguru import logger

from src.agent import Pipeline, PipelineAgentConfig, IOAgent, AgentContext
from src.llm_client import LLMClient
from src.schema.llm_config import LLMConfig


def main():
    with open("configs/openai_gpt_4o_mini_config.json", "r") as f:
        config_dct = json.load(f)
    llm_config = LLMConfig.from_dict(config_dct)
    llm_client = LLMClient(llm_config)

    system_prompmt = """
    You are a highly advanced language model.
    Carefully heed the user's instructions.
    Respond using Markdown.
    """
    pipeline_config = PipelineAgentConfig(
        agent_configs=[
            {
                "agent_type": "IO",
                "agent_id": "first",
                "messages": [
                    {"role": "system", "content": system_prompmt},
                    {"role": "user", "content": "Write {input} emotions in the Ekman classification."},
                ]
            },
            {
                "agent_type": "IO",
                "agent_id": "second",
                "messages": [
                    {"role": "system", "content": system_prompmt},
                    {"role": "user", "content": "You are given a list of emotions and their descriptions."
                                                " Rewrite them as a short list without descriptions. \n{first}"},
                ]
            },
        ],
        edges=[("first", "second")],
        input_id="first",
        output_id="second",
    )

    pipeline = Pipeline(pipeline_config, llm_client)
    context = AgentContext(data={"input": "6"})
    context = pipeline.process(context)
    print(context.get_value(pipeline_config.output_id))
    logger.info(f"Completion tokens: {llm_client.get_output_tokens().sum()}")
    logger.info(f"Prompt tokens: {llm_client.get_input_tokens().sum()}")
    logger.info(f"LLM queries num: {llm_client.get_output_tokens().shape[0]}")
    logger.info(f"Total cost: {llm_client.get_generations_cost():.6f} (usd)")


if __name__ == "__main__":
    main()