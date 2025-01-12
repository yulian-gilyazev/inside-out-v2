from typing import List, Dict, Any
import argparse
import random
import json
from tqdm.auto import trange
import re

from src.llm_client import LLMClient
from src.schema.llm_config import LLMConfig
from src.schema.emotions import Emotion
from src.utils.logger import Logger

""" Example
python3 -m src.scripts.generate_scenarios \
    --n_scenarios 250 \
    --scenario_prompt_path data/prompts/scenario_generation_prompt.txt \
    --scenario_emotion_validation_prompt_path data/prompts/scenario_emotion_validation_prompt.txt \
    --scenario_consistency_validation_prompt_path data/prompts/scenario_consistency_validation_prompt.txt \
    --llm_config_path configs/openai_gpt_4o_config.json \
    --out_path data/synthetic_dialogues/v2/scenarios.json
"""


class ScenariosGenerator:
    N_RETRIES = 3

    GENDERS = [
        "male", "female"
    ]

    AGE_GROUPS = {
        "teen": "teens (13-19 years old)",
        "youth": "youth (19-20 years old)",
        "adult": "adults (30-49 years old)",
        "late middle age": "late middle age (50-64 years old)",
        "senior": "seniors (65 and older)",
    }

    RELATIONSHIPS = {
        "personal": "personal human interaction (i.e. with other people)",
        "social": "social interaction (i.e. with society)"
    }

    MAIN_CHAR_SYNOPSIS_REGEX = r"\*\*Main Character's Scenario Synopsis\*\*: (.*?(?=\*\*Dialogue Partner's Scenario Synopsis\*\*: ))"
    DIALOGUE_PARTNER_SYNOPSIS_REGEX = r"\*\*Dialogue Partner's Scenario Synopsis\*\*: (.*)"

    def __init__(self, llm_client: LLMClient, scenario_prompt_template: str,
                 scenario_emotion_validation_prompt_template: str,
                 scenario_consistency_validation_prompt_template: str,
                 logger: Logger):
        self.llm_client = llm_client
        self.scenario_prompt_template = scenario_prompt_template
        self.scenario_emotion_validation_prompt_template = scenario_emotion_validation_prompt_template
        self.scenario_consistency_validation_prompt_template = scenario_consistency_validation_prompt_template
        self.logger = logger
        self._retries_num = 0

    def generate_scenario(self, emotion: Emotion, relationship: str, gender: str, age_group: str):
        """Generate scenario for given emotion and classification"""
        system_prompt = self.scenario_prompt_template.format(
            emotion=emotion.value, relationship=self.RELATIONSHIPS[relationship],
            age_group=self.AGE_GROUPS[age_group], gender=gender
        )
        messages = [{"role": "system", "content": system_prompt}]
        scenario = self.llm_client.chat(messages).message.content

        main_char_synopsis = re.search(self.MAIN_CHAR_SYNOPSIS_REGEX, scenario, re.DOTALL)
        dialogue_partner_synopsis = re.search(self.DIALOGUE_PARTNER_SYNOPSIS_REGEX, scenario, re.DOTALL)

        if main_char_synopsis is None or dialogue_partner_synopsis is None:
            self.logger.warning("Generated scenario does not match response template")
            return None
        return main_char_synopsis.group(1).strip(), dialogue_partner_synopsis.group(1).strip()

    def validate_scenario(self, main_scenario: str, interlocutor_scenario: str, emotion: Emotion) -> bool:
        """Validate scenario against predefined emotion"""
        prompt = self.scenario_emotion_validation_prompt_template.format(scenario=main_scenario)
        messages = [{"role": "system", "content": prompt}]
        verdict = self.llm_client.chat(messages).message.content
        if emotion.value not in verdict.lower():
            self.logger.warning(f"Emotion validation failed for {emotion.value}. Verdict: {verdict}")
            return False

        prompt = self.scenario_consistency_validation_prompt_template.format(first_scenario=main_scenario,
                                                                             second_scenario=interlocutor_scenario)
        messages = [{"role": "system", "content": prompt}]
        verdict = self.llm_client.chat(messages).message.content
        if "ok" not in verdict.lower():
            self.logger.warning(f"Consistency validation failed. Verdict: {verdict}")
            return False

        return True

    def pipe(self) -> dict | None:
        """Run full scenario pipeline"""
        emotion = random.choice([e for e in Emotion])
        relationship = random.choice(list(self.RELATIONSHIPS.keys()))
        age_group = random.choice(list(self.AGE_GROUPS.keys()))
        gender = random.choice(self.GENDERS)
        scenario = None
        for _ in range(self.N_RETRIES):
            scenario = self.generate_scenario(emotion, relationship, gender, age_group)
            if scenario is None or not self.validate_scenario(scenario[0], scenario[1], emotion):
                scenario = None
                self._retries_num += 1
            else:
                break
        if scenario is not None:
            return {"emotion": emotion.value, "relationship": relationship, "gender": gender,
                    "age_group": age_group, "scenario": {"main": scenario[0], "interlocutor": scenario[1]}}
        else:
            self.logger.error("Failed to generate scenario")
            return None

    def generate(self, n_scenarios: int) -> List[Dict[str, Any]]:
        """Generate scenarios for given number of scenarios"""
        result = []
        self.logger.info(f"Generating {n_scenarios} scenarios")
        for idx in trange(n_scenarios):
            scenario = self.pipe()
            if scenario is not None:
                scenario["id"] = idx
                result.append(scenario)

        self.logger.log(metric_name="given_n_scenarios", value=n_scenarios,
                        log_stdout=True, log_wandb=True)
        self.logger.log(metric_name="generated_scenarios", value=len(result),
                        log_stdout=True, log_wandb=True)
        self.logger.log(metric_name="failed_attempts", value=self._retries_num,
                        log_stdout=True, log_wandb=True)
        self.logger.log(metric_name="generation_cost", value=self.llm_client.get_generations_cost(),
                        log_stdout=True, log_wandb=True)
        self.logger.log(metric_name="input_tokens", value=self.llm_client.get_input_tokens().sum(),
                        log_stdout=True, log_wandb=True)
        self.logger.log(metric_name="output_tokens", value=self.llm_client.get_output_tokens().sum(),
                        log_stdout=True, log_wandb=True)

        return result
    
    @staticmethod
    def save_scenarios(scenarios: List[dict], path):
        with open(path, "w") as f:
            json.dump({"scenarios": scenarios}, f)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_scenarios', type=int, help='Number of Scenarsios to generate')
    parser.add_argument('--out_path', type=str, help='Path where scenarios will be saved')
    parser.add_argument('--scenario_prompt_path', type=str, help='Path to scenario generation prompt')
    parser.add_argument('--scenario_emotion_validation_prompt_path', type=str,
                        help='Path to scenario validation prompt')
    parser.add_argument('--scenario_consistency_validation_prompt_path', type=str,
                        help='Path to scenario consistency validation prompt')
    parser.add_argument('--llm_config_path', type=str, help='Path to llm config',
                        default="llm_config.json")
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    with open(args.scenario_prompt_path, "r") as f:
        scenario_prompt_template = f.read()
    with open(args.scenario_emotion_validation_prompt_path, "r") as f:
        scenario_emotion_validation_prompt_template = f.read()
    with open(args.scenario_consistency_validation_prompt_path, "r") as f:
        scenario_consistency_validation_prompt_template = f.read()
    with open(args.llm_config_path, "r") as f:
        llm_config = LLMConfig.from_dict(json.load(f))

    llm_client = LLMClient(llm_config)

    wandb_config = {"n_scenarios": args.n_scenarios} | {f"llm_{key}": value
                                                        for key, value in llm_config.get_values_dict().items()}

    logger = Logger(
        group="scenarios_generation",
        run_name="synthetic_data_scenarios_generation",
        tags=["v2"],
        config=wandb_config
    )

    scenarios_generator = ScenariosGenerator(
        llm_client, scenario_prompt_template,
        scenario_emotion_validation_prompt_template,
        scenario_consistency_validation_prompt_template,
        logger
    )
    scenarios = scenarios_generator.generate(args.n_scenarios)
    scenarios_generator.save_scenarios(scenarios, args.out_path)


if __name__ == '__main__':
    main()
