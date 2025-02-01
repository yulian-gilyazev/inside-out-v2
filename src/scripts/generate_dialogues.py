from typing import List, Any, Dict
import argparse
import random
import json
from tqdm.auto import tqdm

from src.llm_client import LLMClient, DialogueManager
from src.schema.llm_config import LLMConfig
from src.utils.logger import Logger


""" Example
python3 -m src.scripts.generate_dialogues --n_dialogues_per_scenario 4 \
    --scenarios_path data/synthetic_dialogues/v2/scenarios.json \
    --first_interlocutor_prompt_path data/prompts/first_interlocutor_prompt.txt \
    --second_interlocutor_empathetic_prompt_path data/prompts/second_interlocutor_empathetic_prompt.txt \
    --second_interlocutor_non_empathetic_prompt_path data/prompts/second_interlocutor_non_empathetic_prompt.txt \
    --min_n_rounds 5 \
    --max_n_rounds 7 \
    --llm_config_path configs/openai_gpt_4o_config.json \
    --out_path data/synthetic_dialogues/v2/dialogues.json 
"""


class DialogueGenerator:
    def __init__(self, llm_client: LLMClient, scenarios: List[Any],
                 first_interlocutor_prompt_template: str,
                 second_interlocutor_empathetic_prompt_template: str,
                 second_interlocutor_non_empathetic_prompt_template: str,
                 min_n_rounds: int, max_n_rounds: int,
                 logger: Logger) -> None:
        self.llm_client = llm_client
        self.scenarios = scenarios
        self.first_interlocutor_prompt_template = first_interlocutor_prompt_template
        self.second_interlocutor_empathetic_prompt_template = second_interlocutor_empathetic_prompt_template
        self.second_interlocutor_non_empathetic_prompt_template = second_interlocutor_non_empathetic_prompt_template
        self.min_n_rounds = min_n_rounds
        self.max_n_rounds = max_n_rounds
        self.logger = logger

    def pipe(self, scenario: str, first_system_prompt: str,
             second_system_prompt: str, alt_second_system_prompt: str) -> Dict[str, Any]:

        dialogue_manager = DialogueManager(self.llm_client, self.llm_client,
                                           first_system_prompt=first_system_prompt,
                                           second_system_prompt=second_system_prompt)

        n_rounds = random.randint(self.min_n_rounds, self.max_n_rounds)
        dialogue = dialogue_manager.communicate(n_rounds)

        alt_dialogue_manager = DialogueManager(self.llm_client, self.llm_client,
                                               first_system_prompt=first_system_prompt,
                                               second_system_prompt=alt_second_system_prompt)
        alt_last_utterance = alt_dialogue_manager.append_utterance(dialogue[:-1])

        result = {
            "dialogue": dialogue,
            "alt_last_utterance": alt_last_utterance,
            "emotion": scenario["emotion"],
            "scenario_id": scenario["id"]
        }

        return result

    def generate_dialogues(self, n_dialogues_per_scenario: int) -> List[Dict[str, Any]]:
        dialogues = []
        self.logger.info(f"Generating dialogues, scenarios: {len(self.scenarios)}, dialogues per scenario: {n_dialogues_per_scenario}")
        for scenario_idx, scenario in tqdm(enumerate(self.scenarios)):
            first_system_prompt = self.first_interlocutor_prompt_template.format(
                scenario=scenario["scenario"]["main"],
                emotion=scenario["emotion"]
            )

            second_empathetic_system_prompt = self.second_interlocutor_empathetic_prompt_template.format(
                scenario=scenario["scenario"]["interlocutor"],
                emotion=scenario["emotion"]
            )

            second_non_empathetic_system_prompt = self.second_interlocutor_non_empathetic_prompt_template.format(
                scenario=scenario["scenario"]["interlocutor"],
                emotion=scenario["emotion"]
            )

            for i in range(n_dialogues_per_scenario):
                empathy_label = (i + scenario_idx) % 2
                if empathy_label == 0:
                    result = self.pipe(scenario, first_system_prompt,
                                       second_non_empathetic_system_prompt,
                                       second_empathetic_system_prompt)
                else:
                    result = self.pipe(scenario, first_system_prompt,
                                       second_empathetic_system_prompt,
                                       second_non_empathetic_system_prompt)
                result["empathy_label"] = empathy_label
                result["id"] = len(dialogues)
                result["scenario_id"] = scenario["id"]
                dialogues.append(result)

        self.logger.log(metric_name="generated_dialogues", value=len(dialogues),
                        log_stdout=True, log_wandb=True)
        self.logger.log(metric_name="generation_cost", value=self.llm_client.get_generations_cost(),
                        log_stdout=True, log_wandb=True)
        self.logger.log(metric_name="input_tokens", value=self.llm_client.get_input_tokens().sum(),
                        log_stdout=True, log_wandb=True)
        self.logger.log(metric_name="output_tokens", value=self.llm_client.get_output_tokens().sum(),
                        log_stdout=True, log_wandb=True)
        return dialogues

    @staticmethod
    def save_dialogues(dialogues: List[dict], path):
        with open(path, "w") as f:
            json.dump({"dialogues": dialogues}, f)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_dialogues_per_scenario', type=int, help='Number of dialogues per scenario')
    parser.add_argument('--scenarios_path', type=str, help='Path to scenarios')
    parser.add_argument('--first_interlocutor_prompt_path', type=str, help='Path to first interlocutor prompt')
    parser.add_argument('--second_interlocutor_empathetic_prompt_path', type=str,
                        help='Path to second interlocutor empathetic prompt')
    parser.add_argument('--second_interlocutor_non_empathetic_prompt_path', type=str,
                        help='Path to second interlocutor non-empathetic prompt')
    parser.add_argument('--min_n_rounds', type=int, help='Minimum number of lines in dialogue')
    parser.add_argument('--max_n_rounds', type=int, help='Maximum number of lines in dialogue')
    parser.add_argument('--llm_config_path', type=str, help='Path to llm config')
    parser.add_argument('--out_path', type=str, help='Path where scenarios will be saved')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    
    with open(args.scenarios_path, "r") as f:
        scenarios = json.load(f)["scenarios"]
    with open(args.llm_config_path, "r") as f:
        llm_config = LLMConfig.from_dict(json.load(f))

    with open(args.first_interlocutor_prompt_path, "r") as f:
        first_interlocutor_prompt_template = f.read()
    with open(args.second_interlocutor_empathetic_prompt_path, "r") as f:
        second_interlocutor_empathetic_prompt_template = f.read()

    with open(args.second_interlocutor_non_empathetic_prompt_path, "r") as f:
        second_interlocutor_non_empathetic_prompt_template = f.read()

    llm_client = LLMClient(llm_config)

    wandb_config = (
        {
            "min_n_rounds": args.min_n_rounds, "max_n_rounds": args.max_n_rounds,
            "n_dialogues_per_scenario": args.n_dialogues_per_scenario
        } |
        {
            f"llm_{key}": value for key, value in llm_config.get_values_dict().items()
        }
    )

    logger = Logger(
        group="dialogues_generation",
        run_name="synthetic_data_dialogues_generation",
        tags=["v2"],
        config=wandb_config
    )

    dialogues_generator = DialogueGenerator(
        llm_client, scenarios,
        first_interlocutor_prompt_template,
        second_interlocutor_empathetic_prompt_template,
        second_interlocutor_non_empathetic_prompt_template,
        args.min_n_rounds, args.max_n_rounds,
        logger)

    dialogues = dialogues_generator.generate_dialogues(args.n_dialogues_per_scenario)

    dialogues_generator.save_dialogues(dialogues, args.out_path)


if __name__ == '__main__':
    main()
