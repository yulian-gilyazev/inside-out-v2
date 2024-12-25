from abc import ABC, abstractmethod
from typing import List
import json
from dataclasses import dataclass
from src.schema.emotions import Emotion


@dataclass
class Dialogue:
    first_messages: List[str]
    second_messages: List[str]
    emotion: Emotion = None
    scenario: str = None
    alt_scenario: str = None

    def format_dialogue(self) -> str:
        messages = [f"A: {first}\nB: {second}" for first, second in zip(self.first_messages, self.second_messages)]
        return "\n".join(messages)


class SyntheticEmotionDataset:
    def __init__(self, dialogues_paths: List[str], empathy_labels: List[str], scenario_path: str):
        self.dialogues = []
        self.empathy_labels = []

        with open(scenario_path, "r") as f:
            data = json.load(f)["scenarios"]
            self.scenarios = {}
            for item in data:
                self.scenarios[item["id"]] = item
                
        for dialogue_path, empathy_label in zip(dialogues_paths, empathy_labels):
            with open(dialogue_path, "r") as f:
                data = json.load(f)["dialogues"]
            for dialogue in data:
                scenario_id = dialogue["scenario_id"]
                dialogue = Dialogue(
                    first_messages=[item["content"] for item in dialogue["dialogue"] if item["role"] == "first"],
                    second_messages=[item["content"] for item in dialogue["dialogue"] if item["role"] == "second"],
                    emotion=Emotion.from_str(self.scenarios[scenario_id]["emotion"]),
                    scenario=self.scenarios[scenario_id]["scenario"]["main"],
                    alt_scenario=self.scenarios[scenario_id]["scenario"]["interlocutor"]
                )
                self.dialogues.append(dialogue)
                self.empathy_labels.append(empathy_label)

    def __getitem__(self, idx):
        if idx >= len(self.dialogues):
            raise IndexError
        return self.dialogues[idx], self.empathy_labels[idx]

    def __len__(self):
        return len(self.dialogues)
