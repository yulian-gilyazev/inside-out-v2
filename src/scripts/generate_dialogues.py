from typing import List, Any, Dict
import jsonlines
import argparse
import random
import json
from tqdm.auto import tqdm, trange
from loguru import logger

from src.llm_client import LLMClient, DialogueManager
from src.schema.llm_config import LLMConfig

MAX_N_ROUNDS = 7

""" Example
python3 -m src.scripts.generate_dialogues --n_dialogues_per_scenario 4 \
    --scenarios_path data/scenarios.json \
    --first_interlocutor_prompt_path data/prompts/first_interlocutor_prompt.txt \
    --second_interlocutor_prompt_path data/prompts/second_interlocutor_non_empathetic_prompt.txt \
    --llm_config_path configs/gpt_4o_mini_config.json \
    --out_path data/non_empathetic_dialogues.json 
"""


def generate_dialogues(llm_client: LLMClient,
                       n_dialogues_per_scenario: int,
                       scenarios: List[Dict[str, Any]],
                       first_interlocutor_prompt_template: str,
                       second_interlocutor_prompt_template: str):
    """Generates dialogues from given scenarios"""
    dialogues = []
    logger.info(f"Generating dialogues, scenarios: {len(scenarios)}, dialogues per scenario: {n_dialogues_per_scenario}")
    for scenario in tqdm(scenarios):
        first_system_prompt = first_interlocutor_prompt_template.format(scenario=scenario["scenario"]["main"],
                                                                        emotion=scenario["emotion"])
        second_system_prompt = second_interlocutor_prompt_template.format(scenario=scenario["scenario"]["interlocutor"],
                                                                          emotion=scenario["emotion"])
        dialogue_manager = DialogueManager(llm_client, llm_client,
                                           first_system_prompt=first_system_prompt,
                                           second_system_prompt=second_system_prompt)

        for _ in range(n_dialogues_per_scenario):
            n_rounds = random.randint(4, MAX_N_ROUNDS)
            dialogue = dialogue_manager.communicate(n_rounds)
            dialogues.append({"dialogue": dialogue, 
                              "emotion": scenario["emotion"],
                              "scenario_id": scenario["id"]
                              })
    return dialogues


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_dialogues_per_scenario', type=int, help='Number of dialogues per scenario')
    parser.add_argument('--scenarios_path', type=str, help='Path to scenarios')
    parser.add_argument('--first_interlocutor_prompt_path', type=str, help='Path to first interlocutor prompt')
    parser.add_argument('--second_interlocutor_prompt_path', type=str, help='Path to second interlocutor prompt')
    parser.add_argument('--llm_config_path', type=str, help='Path to llm config')
    parser.add_argument('--out_path', type=str, help='Path where scenarios will be saved')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    
    with open(args.scenarios_path, "r") as f:
        scenarios = json.load(f)["scenarios"]
    with open(args.llm_config_path, "r") as f:
        llm_config = json.load(f)

    with open(args.first_interlocutor_prompt_path, "r") as f:
        first_interlocutor_prompt_template = f.read()
    with open(args.second_interlocutor_prompt_path, "r") as f:
        second_interlocutor_prompt_template = f.read()

    llm_client = LLMClient(LLMConfig.from_dict(llm_config))
    dialogues = generate_dialogues(llm_client, args.n_dialogues_per_scenario, scenarios,
                                   first_interlocutor_prompt_template, second_interlocutor_prompt_template)

    with open(args.out_path, "w") as f:
        json.dump({"dialogues": dialogues}, f)


if __name__ == '__main__':
    main()