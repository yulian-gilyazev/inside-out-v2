from typing import Dict, List, Any
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import more_itertools

from src.schema.emotions import Emotion


class BertERCModel:
    def __init__(self, classifier_model_name: str = "michellejieli/emotion_text_classifier",
                 batch_size: int = 16,
                 dialogue_role: str = "first", id2label: Dict = None):
        self.classifier_model_name = classifier_model_name
        self.batch_size = batch_size
        self.dialogue_role = dialogue_role
        self.classifier = AutoModelForSequenceClassification.from_pretrained(classifier_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(classifier_model_name)
        if id2label is not None:
            self.id2label = id2label
        else:
            self.id2label = {
                0: Emotion.ANGER,
                1: Emotion.DISGUST,
                2: Emotion.FEAR,
                3: Emotion.HAPPINESS,
                4: None,
                5: Emotion.SADNESS,
                6: None
            }

        self.valid_idxs = [i for i, label in self.id2label.items() if label is not None]
        self.valid_labels = [label for i, label in self.id2label.items() if label is not None]

    def _logits_to_probas(self, logits: torch.Tensor) -> List[Dict[str, float]]:
        probas = torch.softmax(logits[:, self.valid_idxs], dim=-1)
        result = [{label.value: value for label, value in zip(self.valid_labels, row)} for row in probas.tolist()]
        return result

    def _format_dialogue(self, dialogue: List[Any]) -> str:
        return "Persons utterances: \n" + "\n".join(dialogue)

    def predict(self, dialogues: List) -> List:
        data = [self._format_dialogue(dialogue) for dialogue in dialogues]
        result = []
        for batch in more_itertools.chunked(data, self.batch_size):
            logits = self.classifier(**self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)).logits
            result.extend(self._logits_to_probas(logits))
        return result
