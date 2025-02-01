import argparse
import json
import more_itertools
from loguru import logger
from tqdm.auto import tqdm

from src.utils.data import SyntheticEmotionDataset, Dialogue, split_dataset
from src.schema.emotions import Emotion
from src.scripts.experiments.gpt_swarm_optimization import *


""" Example
python3 -m src.scripts.experiments.run_gptswarm_erc --out_path 'data/gptswarm_agent_erc_result_gpt4o-mini.json'
"""


class GPTSwarmOptimizedERCAgent:
    edge_probs_path = "models/edge_probs_tensort_final.pt"
    erc_prompt = """
    You feel {emotion}. Act based on what emotion you are experiencing.
    You need to assess emotion of the first (A) interlocutor in the dialogue, estimate your confidence and give reasoning for your answer.
    Your answer should consist of an emotion and an assessment of the level of confidence in it in the range from 0 to 1.
    To select emotions, use Ekman's classification into 5 main emotions - Anger, Disgust, Fear, Happiness, Sadness.
    Separate the emotion and the response using a semicolon.
    Response example:
    `Anger; 0.7`
    """
    edge_prob_threshold = 0.5

    def __init__(self, model_name: str):
        edge_probs = torch.load(self.edge_probs_path)

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

        edge_mask = edge_probs > self.edge_prob_threshold
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
    parser.add_argument('--dialogues_path', type=str,
                        default="data/synthetic_dialogues/v2/dialogues.json", help='Path to dialogues')
    parser.add_argument('--scenarios_path', type=str,
                        default="data/synthetic_dialogues/v2/scenarios.json", help='Path to scenarios')
    parser.add_argument('--model_name', type=str,  default='gpt-4o-mini', help='Name of model')
    parser.add_argument('--out_path', type=str, help='Path where results will be saved')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    dset = SyntheticEmotionDataset(args.dialogues_path, args.scenarios_path)
    dialogues = [dset[i].first_messages for i in range(len(dset))]

    model = GPTSwarmOptimizedERCAgent(model_name=args.model_name)

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