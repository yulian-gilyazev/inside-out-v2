import json
import re
import sys
import time
from typing import Any, Dict, Optional

import torch

import streamlit as st
from src.agent import AgentContext, Pipeline, registry
from src.llm_client import LLMClient
from src.models.bert_erc import BertERCModel
from src.schema.emotions import Emotion
from src.schema.llm_config import LLMConfig
from src.scripts.experiments.gpt_swarm_optimization import *
from src.utils.data import Dialogue

sys.path.append(os.path.join("libs", "GPTSwarm"))
from swarm.graph.swarm import Swarm


class BaseModel:
    @staticmethod
    def _emotion_from_text(text: str) -> Optional[Emotion]:
        try:
            return Emotion.from_str(text.split(";")[0].lower().strip())
        except:
            return None


class InsideOutModel(BaseModel):
    llm_config_path = "configs/gpt_4o_mini_config.json"
    agent_name = "inside-out-erc"

    def __init__(self):

        with open(self.llm_config_path, "r") as f:
            config_dct = json.load(f)
        llm_client = LLMClient(LLMConfig.from_dict(config_dct))

        self.inside_out_pipeline_config = registry.get_config(self.agent_name)

        self.pipeline = Pipeline(self.inside_out_pipeline_config, llm_client)

    def __call__(self, dialogue: Dialogue) -> Optional[Emotion]:

        context = AgentContext(data={"input": dialogue.format_dialogue()})
        context = self.pipeline.process(context)
        predicted = context.get_value(self.inside_out_pipeline_config.output_id)
        emotion = self._emotion_from_text(predicted)
        return emotion


class BertModel(BaseModel):
    def __init__(self):
        self.model = BertERCModel()

    def __call__(self, dialogue: Dialogue) -> Optional[Emotion]:
        predicted = self.model.predict([dialogue.first_messages])[0]
        return max(predicted, key=predicted.get)


class GPTSwarmOptimizedERCAgent(BaseModel):
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

    def __init__(self):
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
            model_name="openai/gpt-4o-mini",
            edge_optimize=True,
        )

        edge_mask = edge_probs > 0.5
        self.realized_graph = self.swarm.connection_dist.realize_mask(
            self.swarm.composite_graph, edge_mask
        )

    def __call__(self, dialogue: Dialogue) -> Optional[Emotion]:
        input_dict = {
            "task": self.erc_prompt + "\nDialogue:\n\n" + dialogue.format_dialogue()
        }

        predicted = self.swarm.run(input_dict, self.realized_graph)
        emotion = self._emotion_from_text(predicted)
        return emotion


def load_models() -> Dict[str, object]:
    models = {
        "Inside Out": InsideOutModel(),
        "BERT Emotion Classifier": BertModel(),
        "GPTSwarm Optimized Agent": GPTSwarmOptimizedERCAgent(),
    }
    return models


def classify_dialogue(model, dialogue: str) -> str:
    """Simulated dialogue classification"""
    time.sleep(1)

    predicted_emotion = model(dialogue)
    if predicted_emotion is None:
        return f"LLM returned answer in unexpected format"

    return f"Classification result: {predicted_emotion}"


def get_example_dialogues() -> Dict[str, str]:
    """Provides examples of dialogues"""
    with open("streamlit/examples.json", "r") as f:
        examples = json.load(f)
    return examples


def get_dialogue_from_text(text: str, first_sep="A", second_sep="B") -> Dialogue:
    pattern = rf"({first_sep}|{second_sep}): (.+)"
    matches = re.findall(pattern, text)

    replicas = {"A": [], "B": []}
    for speaker, text in matches:
        replicas[speaker].append(text)
    first_replicas = replicas[first_sep]
    second_replicas = replicas[second_sep]
    return Dialogue(first_replicas, second_replicas)


def main(models: Dict[str, Any], examples: Dict[str, str]) -> None:
    st.title("Emotion Recognition in Conversation")
    with st.sidebar:
        st.header("About app")
        st.info(
            """
        This is a demo version of the dialogue classifier.

        Features:
        - Dialogue classification  
        - Two input methods  
        """
        )

        st.markdown("---")
        st.subheader("Instruction")
        st.write(
            """
        1. Select a model  
        2. Choose an input method  
        3. Enter or select a dialogue  
        4. Click "Classify" 
        """
        )

    model_name = st.selectbox("Select a classification model 🤖", list(models.keys()))
    selected_model = models[model_name]

    st.markdown("---")

    input_method = st.radio(
        "Dialogue input method 📝", ("Single text", "Separate fields", "Use an example")
    )

    dialogue = ""

    if input_method == "Single text":
        dialogue = st.text_area(
            "Enter the dialogue using identifiers 'A:' and 'B:'",
            height=200,
            placeholder="A: Hi!\nB: Hello!\nA: How are you?\nB: Good, thanks!",
        )
        dialogue = get_dialogue_from_text(dialogue)

    elif input_method == "Separate fields":
        col1, col2 = st.columns(2)
        with col1:
            user_a = st.text_area(
                "\\n separated user A's lines:",
                height=200,
                placeholder="Hi!\nHow are you?",
            )
        with col2:
            user_b = st.text_area(
                "\\n separated user B's lines:",
                height=200,
                placeholder="Hello!\nGood, thanks!",
            )
        user_a = user_a.strip().split("\n")
        user_b = user_b.strip().split("\n")
        dialogue = Dialogue(user_a, user_b)
    else:
        example_choice = st.selectbox(
            "Choose a dialogue example", list(examples.keys())
        )
        dialogue = get_dialogue_from_text(examples[example_choice])
        st.text_area(
            "Dialogue preview:",
            value=dialogue.format_dialogue(),
            height=200,
            disabled=True,
        )

    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("Classify 🔍", use_container_width=True):
            if len(dialogue.first_messages) == 0:
                st.warning("⚠️ Please enter a dialogue for classification.")
            else:
                with st.spinner("Classifying... ⏳"):
                    result = classify_dialogue(selected_model, dialogue)
                    st.success(result)

    with col2:
        if st.button("Clear ❌", use_container_width=True):
            st.rerun()

    with st.expander("📌 Additional Information"):
        st.write(
            """
        ### Model descriptions:
        
        - **Inside Out**: Inside Out Agent.
        - **BERT Emotion Classifier**: michellejieli/emotion_text_classifier
        - **GPTSwarm Optimized Agent**: GPTSwarm agent optimized on emotion classification task
        
        ### Possible results:
        
        - Anger
        - Disgust
        - Fear
        - Happiness
        - Sadness
        """
        )


if __name__ == "__main__":
    models = load_models()
    examples = get_example_dialogues()
    main(models, examples)
