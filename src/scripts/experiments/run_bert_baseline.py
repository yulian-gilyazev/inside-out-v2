import argparse
import json
import os

import more_itertools
from loguru import logger
from tqdm.auto import tqdm

from src.models.bert_erc import BertERCModel
from src.utils.data import EmpatheticDialoguesDataset, SyntheticEmotionDataset

""" Example
python3 -m src.scripts.experiments.run_bert_baseline --dataset 'synthetic' --dataset_path 'data/synthetic_dialogues/v2' --out_path 'data/baseline_erc_bert.json'

python3 -m src.scripts.experiments.run_bert_baseline --dataset 'empatheticdialogues' --dataset_path 'data/empatheticdialogues' --part 'test' --out_path 'data/empatheticdialogues_test_baseline_erc_bert.json'
"""


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent_name', type=str, help='Name of agent')
    parser.add_argument('--dataset', type=str, required=True, choices=["synthetic", "empatheticdialogues"], help='Dataset to use')
    parser.add_argument('--dataset_path', type=str, help='Path to dataset')
    parser.add_argument('--part', type=str, choices=["train", "dev", "test"], required=False, help='Part of dataset to use')
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
        dset = EmpatheticDialoguesDataset(args.dataset_path, args.part)

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