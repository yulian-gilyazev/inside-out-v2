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
* [Synthetic dataset GPTSwarm ERC](https://wandb.ai/yulian-gilyazev/inside-out-v2/runs/fpcdgcor?nw=nwuseryuliangilyazev)
* [Synthetic dataset OPRO ERC](https://wandb.ai/yulian-gilyazev/inside-out-v2/runs/igxatdrq?nw=nwuseryuliangilyazev)
* [Empathetic Dialogues GPTSwarm ERC](https://wandb.ai/yulian-gilyazev/inside-out-v2/runs/kx93b0aj?nw=nwuseryuliangilyazev)
* [Empathetic Dialogues OPRO ERC](https://wandb.ai/yulian-gilyazev/inside-out-v2/runs/aj8pm68s?nw=nwuseryuliangilyazev)

The table above shows the performance comparison of different emotion recognition in conversation (ERC) approaches on our synthetic dialogue dataset:


**Table 1: Performance on Ekman's 5 Basic Emotions**
|   | BERT Baseline| Inside-Out | Inside-Out Generated Emotions | Inside-Out Generated Emotions + Debate| GPTSwarm | GPTSwarm rand |OPRO |
|:---|:-------------|:-------------|:-------------|:-------------|:-------------|:-------------|:-------------|
| synthetic data| 0.645    | 0.735    | 0.745  |  0.73    |   0.8  | 0.625  | 0.755  |
| empathetic dialogues| 0.7673    | 0.8157    | 0.8096   |  -    |   0.78   | -  | -  |


**Table 2: Performance on Extended Emotions Set (32 emotions)**
|   |  | Inside-Out |Inside-Out Generated Emotions |Inside-Out Generated Emotions + Debate| GPTSwarm | GPTSwarm random |
|:---|:-------------|:-------------|:-------------|:-------------|:-------------|:-------------|
| empathetic dialogues | gpt-4o-mini<br>gpt-4o | 0.215<br>0.442 |0.253<br>0.4733|0.266<br>0.4866| 0.233<br>0.4067 | 0.123<br>- |