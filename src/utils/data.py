from typing import List, Tuple
import json
import os
import numpy as np
import pandas as pd
import csv
import copy
from dataclasses import dataclass
from src.schema.emotions import Emotion, EmpatheticDialoguesEmotion
from loguru import logger


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


class EmotionDataset:
    def __init__(self, shuffle=False):
        self.dialogues = []
        self.scenarios = {}
        self._shuffle = shuffle
        self._idxs = []

    @classmethod
    def from_list(cls, dialogues: List[Dialogue],  shuffle: bool = False) -> 'SyntheticEmotionDataset':
        instance = cls.__new__(cls)
        instance.dialogues = dialogues
        instance.scenarios = {}
        instance._shuffle = shuffle
        instance._idxs = np.arange(len(dialogues))
        if shuffle:
            instance.shuffle()
        return instance
            
    def shuffle(self):
        self._shuffle = True
        self._idxs = np.random.permutation(len(self.dialogues))

    def __getitem__(self, _idx):
        idx = self._idxs[_idx]
        if idx >= len(self.dialogues):
            raise IndexError
        return self.dialogues[idx]

    def __len__(self):
        return len(self._idxs)
    
    def __iter__(self):
        return iter(self.dialogues)
    
    def __next__(self):
        if self._idx >= len(self.dialogues):
            raise StopIteration
        item = self.dialogues[self._idx]
        self._idx += 1
        return item

class SyntheticEmotionDataset(EmotionDataset):
    def __init__(self, dialogues_path: str, scenarios_path: str, shuffle=False):
        super().__init__(shuffle=shuffle)

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

        self._idxs = np.arange(len(self.dialogues))

        if shuffle:
            self.shuffle()
    

class EmpatheticDialoguesDataset(EmotionDataset):
    context_to_emotion = {
        'angry': Emotion.ANGER,
        'disgusted': Emotion.DISGUST,
        'afraid': Emotion.FEAR,
        'happy': Emotion.HAPPINESS,
        'sad': Emotion.SADNESS,
    }
    not_extended_emotions = ['angry', 'disgusted', 'afraid', 'happy', 'sad']
    @staticmethod
    def _read_empatheticdialogues_csv(path: str) -> pd.DataFrame:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        headers = lines[0].strip().split(',')
        num_fields = len(headers)
        for line in lines[1:]:
            fields = line.strip().split(',')

        data = []
        
        for line in lines[1:]:
            fields = line.strip().split(',')
            if len(fields) > num_fields:
                fields = fields[:num_fields]
            row = {}
            for header, field in zip(headers, fields):
                if header in ['utterance_idx', 'speaker_idx']:
                    row[header] = int(field)
                elif header in ['prompt', 'utterance']:
                    row[header] = field.replace('_comma_', ',')
                else:
                    row[header] = field
            data.append(row)
            
        return pd.DataFrame(data)

    def __init__(self, dataset_path: str, part: str = "test", extended: bool = False, shuffle=False):
        super().__init__(shuffle=shuffle)
        df = self._read_empatheticdialogues_csv(os.path.join(dataset_path, f"{part}.csv"))
        df.context = df.context.apply(lambda x: x.lower())
        if not extended:
            df = df[df.context.isin(self.not_extended_emotions)]
        for conv_id, group in df.groupby('conv_id'):
            if len(group) < 2 or group["utterance_idx"].nunique() != len(group):
                logger.warning(f"Conversation {conv_id} has repeated or out of order utterances")
                continue
            if group['utterance_idx'].max() != len(group) or group['utterance_idx'].min() != 1:
                logger.warning(f"Conversation {conv_id} has missing utterance indices or out of order")
                continue
            assert group['context'].nunique() == 1, f"Conversation {conv_id} has multiple contexts"
            if not extended:
                emotion = self.context_to_emotion[group.iloc[0]['context']]
            else:
                emotion = EmpatheticDialoguesEmotion.empathy_dialogues_emotion_to_emotion(group.iloc[0]['context'])
            assert group['prompt'].nunique() == 1, f"Conversation {conv_id} has multiple prompts"
            scenario = group.iloc[0]['prompt']
            interlocutor_scenario = ""
            first_messages, second_messages= [], []
            for i, row in group.sort_values(by='utterance_idx').reset_index(drop=True).iterrows():
                assert row['utterance_idx'] == i + 1, f"Conversation {conv_id} has repeated or out of order utterances"
                utterance = row['utterance']
                if i % 2 == 0:
                    first_messages.append(utterance)
                else:
                    second_messages.append(utterance)
            dialogue = Dialogue(
                first_messages=first_messages, 
                second_messages=second_messages,
                emotion=emotion,
                scenario=scenario, 
                interlocutor_scenario=interlocutor_scenario, 
                alt_last_message=None, 
                empathy_label=None,
                id=conv_id,
            )
            self.dialogues.append(dialogue)

        self._idxs = np.arange(len(self.dialogues))
        

def split_dataset(dataset: SyntheticEmotionDataset, size_left: int) -> Tuple[SyntheticEmotionDataset, SyntheticEmotionDataset]:
    assert size_left >= 0
    assert size_left <= len(dataset)

    dataset_left = copy.deepcopy(dataset)
    dataset_right = copy.deepcopy(dataset)

    dataset_left = SyntheticEmotionDataset.from_list(dataset.dialogues[:size_left], shuffle=dataset._shuffle)
    dataset_right = SyntheticEmotionDataset.from_list(dataset.dialogues[size_left:], shuffle=dataset._shuffle)
    
    return dataset_left, dataset_right


def join_datasets(datasets: List[EmotionDataset]) -> EmotionDataset:
    dialogues = []
    for dataset in datasets:
        dialogues.extend(dataset.dialogues)
    return EmotionDataset.from_list(dialogues, shuffle=datasets[0]._shuffle)
