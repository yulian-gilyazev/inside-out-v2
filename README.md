# Readme

## Installation

#todo: switch to poetry
```sh
python3 -m venv env
source env/bin/activate
pip install poetry
pip install -r requirements.txt
cd libs/GPTSwarm && poetry install
python3 -m pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

## Dataset Generation Example
```sh
python3 -m src.scripts.generate_scenarios \
    --n_scenarios <n_scenarios> \
    --scenario_prompt_path data/prompts/scenario_generation_prompt.txt \
    --scenario_emotion_validation_prompt_path data/prompts/scenario_emotion_validation_prompt.txt \
    --scenario_consistency_validation_prompt_path data/prompts/scenario_consistency_validation_prompt.txt \
    --llm_config_path <path_to_llm_config> \
    --out_path <output_path>
```

```sh
python3 -m src.scripts.generate_dialogues --n_dialogues_per_scenario <n_dialogues> \
--scenarios_path <scenarios_path> \
--first_interlocutor_prompt_path data/prompts/first_interlocutor_prompt.txt \
--second_interlocutor_prompt_path data/prompts/second_interlocutor_non_empathetic_prompt.txt \
--llm_config_path <path_to_llm_config> \
--out_path <output_path>
```

