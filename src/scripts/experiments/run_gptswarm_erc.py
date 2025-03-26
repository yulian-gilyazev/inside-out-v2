import argparse
import json
import os
import more_itertools
from loguru import logger
from tqdm.auto import tqdm

from src.utils.data import SyntheticEmotionDataset, EmpatheticDialoguesDataset, split_dataset, Dialogue
from src.schema.emotions import Emotion
from src.scripts.experiments.gpt_swarm_erc_optimization import *


""" Example
python3 -m src.scripts.experiments.run_gptswarm_erc --dataset 'synthetic' --dataset_path 'data/synthetic_dialogues/v2' --out_path 'data/gptswarm_agent_erc_exp2_result_gpt4o.json' --model_name 'gpt-4o'

python3 -m src.scripts.experiments.run_gptswarm_erc --dataset 'empatheticdialogues' --dataset_path 'data/empatheticdialogues' --part 'test' --out_path 'data/empatheticdialogues_test_gptswarm_erc_exp2_result_gpt4o-mini.json'

python3 -m src.scripts.experiments.run_gptswarm_erc --dataset 'empatheticdialogues' --dataset_path 'data/empatheticdialogues' --part 'test' --emotions_set 'extended' --out_path 'data/empatheticdialogues_test_gptswarm_erc_exp2_result_gpt4o-mini_extended.json'
"""


class GPTSwarmOptimizedERCAgent:

    def __init__(self, erc_prompt: str, model_name: str, edge_prob_threshold: float = 0.5, edge_probs_path: str = "models/gptswarm_erc_edge_probs_tensor.pt"):
        edge_probs = torch.load(edge_probs_path)
        self.erc_prompt = erc_prompt

        self.swarm = Swarm(
            [
                "AngerERCCOT",
                "DisgustERCCOT",
                "FearERCCOT",
                "HappinessERCCOT",
                "SadnessERCCOT",
            ],
            "gaia",
            model_name=model_name,
            edge_optimize=True,
        )

        edge_mask = edge_probs > edge_prob_threshold
        self.realized_graph = self.swarm.connection_dist.realize_mask(
            self.swarm.composite_graph, edge_mask
        )

    def __call__(self, dialogue: Dialogue) -> str:
        input_dict = {
            "task": self.erc_prompt + "\nDialogue:\n\n" + dialogue.format_dialogue()
        }

        predicted = self.swarm.run(input_dict, self.realized_graph)[0]
        return predicted


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_name', type=str, help='Name of agent')
    parser.add_argument('--dataset', type=str, required=True, choices=["synthetic", "empatheticdialogues"], help='Dataset to use')
    parser.add_argument('--dataset_path', type=str, help='Path to dataset')
    parser.add_argument('--part', type=str, choices=["train", "dev", "test"], required=False, help='Part of dataset to use')
    parser.add_argument('--model_name', type=str,  default='gpt-4o-mini', help='Namse of model')
    parser.add_argument('--emotions_set', type=str, choices=["base", "extended"], default="base", help='Emotions set to use')
    parser.add_argument('--edge_prob_threshold', type=float, default=0.5, help='Edge probability threshold')
    parser.add_argument('--path_to_edge_probs', type=str, default="models/gptswarm_erc_edge_probs_tensor.pt", help='Path to edge probabilities')
    parser.add_argument('--test_size', type=int, default=200, help='Test size')
    parser.add_argument('--out_path', type=str, help='Path where results will be saved')
    args = parser.parse_args()
    if args.dataset == "empatheticdialogues":
        assert args.part is not None, "Part must be specified for synthetic dataset"
    return args


def main():
    args = parse_arguments()

    if args.dataset == "synthetic":
        dialogues_path = os.path.join(args.dataset_path, "dialogues.json")
        scenarios_path = os.path.join(args.dataset_path, "scenarios.json")
        dset = SyntheticEmotionDataset(dialogues_path, scenarios_path)
    elif args.dataset == "empatheticdialogues":
        dset = EmpatheticDialoguesDataset(args.dataset_path, args.part, extended=args.emotions_set == "extended")
    
    dset, _ = split_dataset(dset, args.test_size)

    dialogues = [dset[i].first_messages for i in range(len(dset))]

    if args.emotions_set == "base":
        emotions_cls = Emotion
    elif args.emotions_set == "extended":
        emotions_cls = EmpatheticDialoguesEmotion
      
    emotions_list = [emotion.lower().capitalize() for emotion in emotions_cls.__members__.keys()]
    emotions_list_str = ", ".join(emotions_list)

    n_emotions = len(emotions_list)

    erc_prompt = f"""
You feel {{emotion}}. Act based on what emotion you are experiencing.
You need to assess emotion of the first (A) interlocutor in the dialogue, estimate your confidence and give reasoning for your answer.
Your answer should consist of an emotion and an assessment of the level of confidence in it in the range from 0 to 1.
To select emotions, use classification into {n_emotions} emotions - {emotions_list_str}.
Separate the emotion and the response using a semicolon.
Response example:
`Anger; 0.7`
    """

    model = GPTSwarmOptimizedERCAgent(erc_prompt=erc_prompt, model_name=args.model_name)

    logger.info(f"Start inference on {len(dialogues)} dialogues")
    result = []
    for i in tqdm(range(len(dset))):
        dialogue = dset[i]
        prediction = model(dialogue)
        result.append({"id": dset[i].id,  "empathy_label": dset[i].empathy_label, "prediction": prediction})

    with open(args.out_path, "w") as f:
        json.dump({"predictions": result}, f)
    logger.info(f"Saved results to {args.out_path}")


if __name__ == "__main__":
    main()