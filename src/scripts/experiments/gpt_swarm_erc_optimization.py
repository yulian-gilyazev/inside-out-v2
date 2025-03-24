import sys
import os
import asyncio
from typing import Optional, List, Any, Dict, Literal
from tqdm import tqdm
import torch
import time
import numpy as np
from dataclasses import dataclass, asdict
from collections import Counter

from src.utils.data import SyntheticEmotionDataset, EmpatheticDialoguesDataset, split_dataset
from src.utils.logger import Logger
from src.schema.emotions import Emotion, EmpatheticDialoguesEmotion

sys.path.append(os.path.join("libs", "GPTSwarm"))

from swarm.graph import Node, Graph
from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry
from swarm.llm.format import Message
from swarm.llm import LLMRegistry
from swarm.environment.agents.agent_registry import AgentRegistry

from swarm.graph import Graph
from swarm.graph.swarm import Swarm


class ERCCoTStep(Node):
    def __init__(
        self,
        domain: str,
        model_name: Optional[str],
        is_last_step: bool,
        operation_description: str = "Make one step of CoT",
        emotion: Optional[str] = None,
        id=None,
    ):
        super().__init__(operation_description, id, True)
        self.domain = domain
        self.model_name = model_name
        self.is_last_step = is_last_step
        self.emotion = emotion
        self.llm = LLMRegistry.get(model_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role()
        self.constraint = self.prompt_set.get_constraint()

    @property
    def node_name(self):
        return self.__class__.__name__

    async def _execute(self, inputs: List[Any] = [], **kwargs):

        node_inputs = self.process_input(inputs)
        outputs = []
        for input_dict in node_inputs:

            role = self.prompt_set.get_role()
            constraint = self.prompt_set.get_constraint()
            if self.is_last_step:
                system_prompt = (
                    f"You are {role}. {constraint}. "
                    "Answer taking into consideration the provided sequence "
                    "of thoughts on the question at hand."
                )
                max_tokens = 25
            else:
                system_prompt = (
                    f"You are {role}. "
                    "Given the question, solve it step by step. "
                    "Answer your thoughts about the next step of the solution given "
                    "everything that has been provided to you so far. "
                    "Expand on the next step. "
                    "Do not try to provide the answer straight away, instead expand "
                    "on your thoughts about the next step of the solution."
                    "Aswer in maximum 30 words. "
                    "Do not expect additional input. Make best use of whatever "
                    "knowledge you have been already provided."
                )
                max_tokens = 80
            if "output" in input_dict:
                task = input_dict["output"]
            else:
                task = input_dict["task"].format(emotion=self.emotion)
            user_prompt = self.prompt_set.get_answer_prompt(question=task)
            message = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt),
            ]
            response = await self.llm.agen(message, max_tokens=max_tokens)
            if self.is_last_step:
                concatenated_response = response
            else:
                concatenated_response = (
                    f"{task}. Here is the next thought. {response}. "
                )

            execution = {
                "operation": self.node_name,
                "task": task,
                "files": input_dict.get("files", []),
                "input": task,
                "role": role,
                "constraint": constraint,
                "prompt": user_prompt,
                "output": concatenated_response,
                "ground_truth": input_dict.get("GT", []),
                "format": "natural language",
            }
            outputs.append(execution)
            self.memory.add(self.id, execution)

        return outputs



class ERCCOT(Graph):
    emotion = None

    def build_graph(self):
        num_thoughts = 2
        thoughts = []
        for i_thought in range(num_thoughts):
            thought = ERCCoTStep(
                self.domain,
                self.model_name,
                is_last_step=i_thought == num_thoughts - 1,
                emotion=self.emotion,
            )
            if i_thought > 0:
                thoughts[-1].add_successor(thought)
            thoughts.append(thought)

        self.input_nodes = [thoughts[0]]
        self.output_nodes = [thoughts[-1]]

        for thought in thoughts:
            self.add_node(thought)


@AgentRegistry.register("AngerERCCOT")
class AngerERCCOT(ERCCOT):
    emotion = "anger"


@AgentRegistry.register("DisgustERCCOT")
class DisgustERCCOT(ERCCOT):
    emotion = "disgust"


@AgentRegistry.register("FearERCCOT")
class FearERCCOT(ERCCOT):
    emotion = "fear"


@AgentRegistry.register("HappinessERCCOT")
class HappinessERCCOT(ERCCOT):
    emotion = "happiness"


@AgentRegistry.register("SadnessERCCOT")
class SadnessERCCOT(ERCCOT):
    emotion = "sadness"


@dataclass
class OptimizationConfig:
    lr: float
    num_iters: int
    batch_size: int
    task_prompt: str

    def to_dict(self) -> Dict[str, Any]:
        return {k: str(v) for k, v in asdict(self).items()}


class Optimizer:
    evaluation_n_steps = 60

    def __init__(
        self, swarm, train_dataset: SyntheticEmotionDataset,
            test_dset: SyntheticEmotionDataset, config: OptimizationConfig, logger: Logger
    ) -> None:

        self._swarm: Optional[Swarm] = swarm
        self._train_dataset = train_dataset
        self._test_dset = test_dset
        self.config = config
        self.logger = logger

    def evaluate(self):
        assert self._swarm is not None

        dataset = self._test_dset

        self.logger.info("Evaluating")

        start_ts = time.time()

        future_answers = []
        log_probs = []
        correct_answers = []

        for i in range(len(dataset)):
            record = dataset[i]

            realized_graph, log_prob = self._swarm.connection_dist.realize(
                self._swarm.composite_graph,
            )

            input_dict = {
                "task": self.config.task_prompt + "\nDialogue:\n\n" + record.format_dialogue()
            }
            answer = self._swarm.arun(input_dict, realized_graph)
            future_answers.append(answer)
            log_probs.append(log_prob)
            correct_answer = record.emotion.value
            correct_answers.append(correct_answer)

        async def run_coroutines():
            return await asyncio.gather(*future_answers)

        raw_answers = asyncio.run(run_coroutines())

        self.logger.info(f"Inference time {time.time() - start_ts:.3f}")

        utilities: List[float] = []
        for raw_answer, log_prob, correct_answer in zip(
                raw_answers, log_probs, correct_answers
        ):
            self.logger.info(raw_answer)
            answer = raw_answer[0].split(";")[0].lower()
            assert isinstance(
                correct_answer, str
            ), f"String expected but got {correct_answer} of type {type(correct_answer)} (1)"
            utility = answer == correct_answer
            utilities.append(utility)

        self.logger.info(f"utilities: {np.mean(np.array(utilities))}")

        mean_utility = np.mean(np.array(utilities))

        self.logger.log(
            metric_name="val_utility",
            value=mean_utility.item(),
            log_stdout=True,
            log_wandb=True,
        )
        self.logger.info("Done!")

    def optimize_swarm(self) -> torch.Tensor:

        assert self._swarm is not None

        dataset = self._train_dataset

        self.logger.info("Optimizing swarm")

        optimizer = torch.optim.Adam(
            self._swarm.connection_dist.parameters(), lr=self.config.lr
        )
        len_dataset = len(dataset)
        for i_iter in range(self.config.num_iters):
            if i_iter % self.evaluation_n_steps == 0:
                self.evaluate()
            self.logger.info(f"Iter {i_iter}\n{80 * '-'}")

            start_ts = time.time()

            future_answers = []
            log_probs = []
            correct_answers = []

            for i_record in tqdm(range(self.config.batch_size)):
                i_record = (self.config.batch_size * i_iter + i_record) % len_dataset
                record = dataset[i_record]

                realized_graph, log_prob = self._swarm.connection_dist.realize(
                    self._swarm.composite_graph,
                )

                input_dict = {
                    "task": self.config.task_prompt + "\nDialogue:\n\n" + record.format_dialogue()
                }
                answer = self._swarm.arun(input_dict, realized_graph)
                future_answers.append(answer)
                log_probs.append(log_prob)
                correct_answer = record.emotion.value
                correct_answers.append(correct_answer)

            async def run_coroutines():
                return await asyncio.gather(*future_answers)

            raw_answers = asyncio.run(run_coroutines())

            self.logger.info(f"Batch time {time.time() - start_ts:.3f}")

            loss_list: List[torch.Tensor] = []
            utilities: List[float] = []
            for raw_answer, log_prob, correct_answer in zip(
                raw_answers, log_probs, correct_answers
            ):
                self.logger.info(raw_answer)
                answer = raw_answer[0].split(";")[0].lower()
                assert isinstance(
                    correct_answer, str
                ), f"String expected but got {correct_answer} of type {type(correct_answer)} (1)"
                utility = answer == correct_answer
                utilities.append(utility)
                single_loss = -log_prob * utility
                loss_list.append(single_loss)

            self.logger.info(f"utilities: {utilities}")

            mean_utility = np.mean(np.array(utilities))
            total_loss = torch.mean(torch.stack(loss_list))

            optimizer.zero_grad()
            total_loss.backward()
            self.logger.log(
                metric_name="grad",
                value=self._swarm.connection_dist.edge_logits.grad,
                log_stdout=True,
                log_wandb=False,
            )

            optimizer.step()

            self.logger.log(
                metric_name="edge_logits",
                value=self._swarm.connection_dist.edge_logits,
                log_stdout=True,
                log_wandb=False,
            )

            edge_probs = torch.sigmoid(self._swarm.connection_dist.edge_logits)

            self.logger.log(
                metric_name="edge_probs",
                value=edge_probs,
                log_stdout=True,
                log_wandb=False,
            )

            tensor_path = "edge_probs_tensor.pt"
            torch.save(edge_probs, tensor_path)

            artifact = self.logger.wandb.Artifact(name=f"edge_probs_{i_iter}", type="dataset")
            artifact.add_file(tensor_path)
            self.logger.run.log_artifact(artifact)

            self.logger.log(
                metric_name="train_loss",
                value=total_loss.item(),
                log_stdout=True,
                log_wandb=True,
            )

            self.logger.log(
                metric_name="train_utility",
                value=mean_utility.item(),
                log_stdout=True,
                log_wandb=True,
            )

            self.logger.info("end of iteration")

        self.logger.info("Done!")
        edge_probs = torch.sigmoid(self._swarm.connection_dist.edge_logits)
        return edge_probs
    

@dataclass
class TrainConfig:
    lr: float = 0.1
    num_iters: int = 482
    batch_size: int = 32
    dataset: Literal["empatheticdialogues", "synthetic"] = "empatheticdialogues"
    dataset_path: str = "data/empatheticdialogues"
    emotions_set: Literal["base", "extended"] = "extended"
    test_size: int = 300
    model_name: str = "gpt-4o-mini"

    def to_dict(self) -> Dict[str, Any]:
        return {k: str(v) for k, v in asdict(self).items()}
    


# @dataclass
# class TrainConfig:
#     lr: float = 0.2
#     num_iters: int = 4
#     batch_size: int = 8
#     dataset: Literal["empatheticdialogues", "synthetic"] = "empatheticdialogues"
#     dataset_path: str = "data/empatheticdialogues"
#     emotions_set: Literal["base", "extended"] = "extended"
#     test_size: int = 8
#     model_name: str = "gpt-4o-mini"

#     def to_dict(self) -> Dict[str, Any]:
#         return {k: str(v) for k, v in asdict(self).items()}


def main(config: TrainConfig):
    logger = Logger(
        group="gptswarm_erc",
        run_name="inside-out-cot-agents-empatheticdialogues-extended-03-23",
        tags=["inside-out-cot-agents", config.dataset, config.emotions_set],
        config=config.to_dict(),
        use_wandb=True,
    )

    if config.emotions_set == "base":
        emotions_cls = Emotion
    elif config.emotions_set == "extended":
        emotions_cls = EmpatheticDialoguesEmotion
    else:
        raise ValueError(f"Invalid emotions set: {config.emotions_set}")
    
    emotions_list = [emotion.lower().capitalize() for emotion in emotions_cls.__members__.keys()]
    emotions_list_str = ", ".join(emotions_list)

    n_emotions = len(emotions_list)

    logger.info(f"Emotions list: {emotions_list_str}")

    logger.info(f"N emotions: {n_emotions}")


    erc_prompt = f"""
    You feel {{emotion}}. Act based on what emotion you are experiencing.
    You need to assess emotion of the first (A) interlocutor in the dialogue, estimate your confidence and give reasoning for your answer.
    Your answer should consist of an emotion and an assessment of the level of confidence in it in the range from 0 to 1.
    To select emotions, use classification into {n_emotions} emotions - {emotions_list_str}.
    Separate the emotion and the response using a semicolon.
    Response example:
    `Anger; 0.7`
    """
    logger.info(erc_prompt)

    if config.dataset == "empatheticdialogues":
        train_dset = EmpatheticDialoguesDataset(config.dataset_path, part="train", extended=(config.emotions_set == "extended"))
        test_dset = EmpatheticDialoguesDataset(config.dataset_path, part="test", extended=(config.emotions_set == "extended"))
        test_dset, _ = split_dataset(test_dset, config.test_size)
    elif config.dataset == "synthetic":
        scenarios_path = os.path.join(config.dataset_path, "scenarios.json")
        dialogues_path = os.path.join(config.dataset_path, "dialogues.json")
        dset = SyntheticEmotionDataset(dialogues_path, scenarios_path)
        test_dset, train_dset = split_dataset(dset, config.test_size)
        train_dset.shuffle()
    else:
        raise ValueError(f"Invalid dataset: {config.dataset}")
    
    logger.info(f"Emotions in train: {Counter([d.emotion for d in train_dset])}")
    logger.info(f"N emotions in train: {len(set([d.emotion for d in train_dset]))}")
    
    training_config = OptimizationConfig(
        lr=config.lr,
        num_iters=config.num_iters,
        batch_size=config.batch_size,
        task_prompt=erc_prompt,
    )

    swarm = Swarm(
        [
            "AngerERCCOT",
            "DisgustERCCOT",
            "FearERCCOT",
            "HappinessERCCOT",
            "SadnessERCCOT",
        ],
        "gaia",
        model_name=config.model_name,
        edge_optimize=True,
    )

    optimizer = Optimizer(swarm, train_dset, test_dset, training_config, logger)
    optimizer.optimize_swarm()


if __name__ == "__main__":
    os.environ["GPTSWARM_API_URL"] = "https://api.openai.com/v1"

    main(TrainConfig())
