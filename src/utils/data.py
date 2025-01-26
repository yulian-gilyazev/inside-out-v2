from typing import List
import json
import numpy as np
import copy
from dataclasses import dataclass
from src.schema.emotions import Emotion


@dataclass
class Dialogue:
    first_messages: List[str]
    second_messages: List[str]
    emotion: Emotion = None
    scenario: str = None
    interlocutor_scenario: str = None
    alt_last_message: str = None
    empathy_label: int = None
    id: int = None

    def format_dialogue(self) -> str:
        messages = [f"A: {first}\nB: {second}" for first, second in zip(self.first_messages, self.second_messages)]
        return "\n".join(messages)


class SyntheticEmotionDataset:
    def __init__(self, dialogues_path: str, scenarios_path: str, shuffle=False):
        self.shuffle = shuffle

        self.dialogues = []

        with open(scenarios_path, "r") as f:
            data = json.load(f)["scenarios"]
            self.scenarios = {}
            for item in data:
                self.scenarios[item["id"]] = item
                
        with open(dialogues_path, "r") as f:
            data = json.load(f)["dialogues"]

        for data_item in data:
            scenario_id = data_item["scenario_id"]
            dialogue = Dialogue(
                first_messages=[item["content"] for item in data_item["dialogue"] if item["role"] == "first"],
                second_messages=[item["content"] for item in data_item["dialogue"] if item["role"] == "second"],
                emotion=Emotion.from_str(self.scenarios[scenario_id]["emotion"]),
                scenario=self.scenarios[scenario_id]["scenario"]["main"],
                interlocutor_scenario=self.scenarios[scenario_id]["scenario"]["interlocutor"],
                alt_last_message=data_item["alt_last_utterance"],
                empathy_label=data_item["empathy_label"],
                id=data_item["id"]
            )
            self.dialogues.append(dialogue)

        if self.shuffle:
            self._idxs = np.random.permutation(len(self.dialogues))
        else:
            self._idxs = np.arange(len(self.dialogues))

    def __getitem__(self, _idx):
        idx = self._idxs[_idx]
        if idx >= len(self.dialogues):
            raise IndexError
        return self.dialogues[idx]

    def __len__(self):
        return len(self._idxs)


def split_dataset(dataset: SyntheticEmotionDataset, size: int):
    dataset = copy.deepcopy(dataset)
    dataset._idxs = dataset._idxs[:size]
    return dataset
