import sys
import os
import torch
from loguru import logger
sys.path.append(os.path.join("libs", "GPTSwarm"))

from swarm.graph.swarm import Swarm
from src.scripts.experiments.gpt_swarm_optimization import *

from src.config import VSEGPT_API_KEY


def main():
    edge_probs = torch.load("models/edge_probs_tensort_final.pt")
    print(edge_probs.shape)
    print(edge_probs)

    logger.info("Initializing swarm")

    swarm = Swarm(
        [
            "AngerERCCOT",
            "DisgustERCCOT",
            "FearERCCOT",
            "HappinessERCCOT",
            "SadnessERCCOT",
        ],
        "gaia",
        model_name="openai/gpt-4o-mini",
        edge_optimize=True,
    )

    edge_mask = edge_probs > 0.5
    realized_graph = swarm.connection_dist.realize_mask(swarm.composite_graph, edge_mask)

    dset = SyntheticEmotionDataset("data/synthetic_dialogues/v2/dialogues.json",
                                   "data/synthetic_dialogues/v2/scenarios.json")
    item = dset[0]

    input_dict = {
        "task": erc_prompt + "\nDialogue:\n\n" + item.format_dialogue()
    }
    print('here')

    answer = swarm.run(input_dict, realized_graph)

    logger.info(f"Answer: {answer}")


if __name__ == "__main__":
    # "https://api.openai.com/v1", "https://api.vsegpt.ru/v1"
    os.environ["GPTSWARM_API_URL"] = "https://api.openai.com/v1"
    main()