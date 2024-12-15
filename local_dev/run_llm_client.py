import json

from src.llm_client import LLMClient
from src.schema.llm_config import LLMConfig


def main():
    with open("configs/gpt_4o_mini_config.json", "r") as f:
        llm_config = LLMConfig.from_dict(json.load(f))
    llm_client = LLMClient(llm_config)
    system_prompmt = """
    You are a highly advanced language model.
    Carefully heed the user's instructions.
    Respond using Markdown.
    """
    messages = [
        {"role": "system", "content": system_prompmt},
        {"role": "user", "content": "Write 5 emotions in the Ekman classification."},
    ]
    response = llm_client.chat(messages).message.content
    print(response)
    print(f"input tokens: {llm_client.get_input_tokens()}")
    print(f"output tokens: {llm_client.get_output_tokens()}")
    print(f"total tokens: {llm_client.get_total_tokens()}")


if __name__ == '__main__':
    main()