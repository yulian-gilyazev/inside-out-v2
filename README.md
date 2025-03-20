# Inside Out V2 Emotion Recognition and Empathetic Generation System

## Overview

The repo consists of:
- Dataset generation pipeline for creating synthetic emotional conversations
- Emotion recognition agents specialized for different emotions (Anger, Disgust, Fear, Happiness, Sadness)
- Epathetic response generation agents (in progress)
- Streamlit web interface for interactive usage

## Quick Start

### Requirements
- Python 3.10+
- OpenAI API key
- Docker (optional, for running the web interface)


### Installation

```sh
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
echo -e "OPENAI_API_KEY=<your_openai_key>" > .env
```


### Dataset Generation

```sh
python3 -m src.scripts.generate_scenarios \
    --n_scenarios <n_scenarios> \
    --scenario_prompt_path configs/prompts/scenario_generation_prompt.txt \
    --scenario_emotion_validation_prompt_path configs/prompts/scenario_emotion_validation_prompt.txt \
    --scenario_consistency_validation_prompt_path configs/prompts/scenario_consistency_validation_prompt.txt \
    --llm_config_path <llm_config_path> \
    --out_path <scenarios_path>
```

```sh
python3 -m src.scripts.generate_dialogues --n_dialogues_per_scenario <n_dialogues_per_scenario> \
    --scenarios_path <scenarios_path> \
    --first_interlocutor_prompt_path configs/prompts/first_interlocutor_prompt.txt \
    --second_interlocutor_empathetic_prompt_path configs/prompts/second_interlocutor_empathetic_prompt.txt \
    --second_interlocutor_non_empathetic_prompt_path configs/prompts/second_interlocutor_non_empathetic_prompt.txt \
    --min_n_rounds <min_n_rounds> \
    --max_n_rounds <min_n_rounds> \
    --llm_config_path <llm_config_path> \
    --out_path <dialogues_path>
```


### ERC Task GPTSwarm optimization
```sh
python3 -m src.scripts.experiments.gpt_swarm_optimization
```


## Results

Exp logs:
* [GPTSwarm ERC](https://wandb.ai/yulian-gilyazev/inside-out-v2/runs/fpcdgcor?nw=nwuseryuliangilyazev)


The table above shows the performance comparison of different emotion recognition in conversation (ERC) approaches on our synthetic dialogue dataset:


|   | BERT Baseline| Inside-Out | Inside-Out Generated Emotions | Inside-Out Generated Emotions + Debate| GPTSwarm | GPTSwarm random |
|:---|:-------------|:-------------|:-------------|:-------------|:-------------|:-------------|
| synthetic data| 0.645    | 0.735    | 0.745  |  0.73    |    0.8   | 0.625   |
| empathetic dialogues| 0.7673    | 0.8157    | 0.8096   |  -    |   -   | -   |