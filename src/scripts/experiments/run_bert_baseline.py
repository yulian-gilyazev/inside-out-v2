import argparse
import json
import more_itertools
from loguru import logger
from tqdm.auto import tqdm

from src.utils.data import SyntheticEmotionDataset
from src.models.bert_erc import BertERCModel

""" Example
python3 -m src.scripts.experiments.run_bert_baseline --out_path 'data/baseline_erc_bert.json'
"""


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_name', type=str, help='Name of agent')
    parser.add_argument('--dialogues_path', type=str,
                        default="data/synthetic_dialogues/v2/dialogues.json", help='Path to dialogues')
    parser.add_argument('--scenarios_path', type=str,
                        default="data/synthetic_dialogues/v2/scenarios.json", help='Path to scenarios')
    parser.add_argument('--out_path', type=str, help='Path where results will be saved')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    dset = SyntheticEmotionDataset(args.dialogues_path, args.scenarios_path)
    dialogues = [dset[i].first_messages for i in range(len(dset))]

    model = BertERCModel()

    logger.info(f"Start inference on {len(dialogues)} dialogues")
    predictions = []
    for dialogues_batch in tqdm(more_itertools.chunked(dialogues, 32)):
        predictions.extend(model.predict(dialogues_batch))
    result = []
    for i in range(len(dset)):
        curr_item = {"id": dset[i].id, "empathy_label": dset[i].empathy_label, "prediction": predictions[i]}
        result.append(curr_item)

    with open(args.out_path, "w") as f:
        json.dump({"predictions": result}, f)
    logger.info(f"Saved results to {args.out_path}")


if __name__ == "__main__":
    main()