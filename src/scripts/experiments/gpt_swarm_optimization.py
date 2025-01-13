import sys
import os
import asyncio
from typing import Optional, List, Any
from tqdm import tqdm
import torch
import time
import numpy as np

from src.utils.data import SyntheticEmotionDataset
from src.utils.logger import Logger

sys.path.append(os.path.join("libs", "GPTSwarm"))

from swarm.graph import Node, Graph
from swarm.environment.prompt.prompt_set_registry import PromptSetRegistry
from swarm.llm.format import Message
from swarm.llm import LLMRegistry
from swarm.environment.operations.cot_step import CoTStep
from swarm.environment.agents.agent_registry import AgentRegistry

from swarm.graph import Graph
from swarm.graph.swarm import Swarm


class CoTStep(Node):
    def __init__(self,
                 domain: str,
                 model_name: Optional[str],
                 is_last_step: bool,
                 operation_description: str = "Make one step of CoT",
                 id=None):
        super().__init__(operation_description, id, True)
        self.domain = domain
        self.model_name = model_name
        self.is_last_step = is_last_step
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
                    "of thoughts on the question at hand.")
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
                    "knowledge you have been already provided.")
            if 'output' in input_dict:
                task = input_dict['output']
            else:
                task = input_dict["task"]
            user_prompt = self.prompt_set.get_answer_prompt(question=task)
            message = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt)
            ]
            response = await self.llm.agen(message, max_tokens=50)
            if self.is_last_step:
                concatenated_response = response
            else:
                concatenated_response = f"{task}. Here is the next thought. {response}. "

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
                "format": "natural language"
            }
            outputs.append(execution)
            self.memory.add(self.id, execution)

        return outputs

erc_prompt = """
You need to assess emotion of the first (A) interlocutor in the dialogue, estimate your confidence and give reasoning for your answer.
Your answer should consist of an emotion and an assessment of the level of confidence in it in the range from 0 to 1.
To select emotions, use Ekman's classification into 5 main emotions - Anger, Disgust, Fear, Happiness, Sadness.
Separate the emotion and the response using a semicolon.
Response example:
`Anger; 0.7`
"""

@AgentRegistry.register('InsideOutCOT')
class CustomCOT(Graph):
    def build_graph(self):

        num_thoughts = 3

        assert num_thoughts >= 2

        thoughts = []
        for i_thought in range(num_thoughts):
            thought = CoTStep(self.domain,
                              self.model_name,
                              is_last_step=i_thought==num_thoughts-1)
            if i_thought > 0:
                thoughts[-1].add_successor(thought)
            thoughts.append(thought)

        self.input_nodes = [thoughts[0]]
        self.output_nodes = [thoughts[-1]]

        for thought in thoughts:
            self.add_node(thought)


class Evaluator():
    def __init__(
            self,
            swarm,
            train_dataset,
            logger: Logger = None
    ) -> None:

        self._swarm: Optional[Swarm] = swarm
        self._train_dataset = train_dataset
        self.logger = logger

    def optimize_swarm(
            self,
            num_iters: int,
            lr: float,
    ) -> torch.Tensor:

        assert self._swarm is not None

        dataset = self._train_dataset

        print(f"Optimizing swarm")

        optimizer = torch.optim.Adam(self._swarm.connection_dist.parameters(), lr=lr)
        batch_size = 8
        len_dataset = 64
        for i_iter in range(num_iters):
            print(f"Iter {i_iter}", 80*'-')

            start_ts = time.time()

            future_answers = []
            log_probs = []
            correct_answers = []

            for i_record in tqdm(range(batch_size)):
                i_record = (batch_size * i_iter + i_record) % len_dataset
                record = dataset[i_record]

                realized_graph, log_prob = self._swarm.connection_dist.realize(
                    self._swarm.composite_graph,
                )

                input_dict = {"task": erc_prompt + "\nDialogue:\n\n" + record.format_dialogue()}
                answer = self._swarm.arun(input_dict, realized_graph)
                future_answers.append(answer)
                log_probs.append(log_prob)
                correct_answer = record.emotion.value
                correct_answers.append(correct_answer)

            async def run_coroutines():
                return await asyncio.gather(*future_answers)

            raw_answers = asyncio.run(run_coroutines())

            print(f"Batch time {time.time() - start_ts:.3f}")

            loss_list: List[torch.Tensor] = []
            utilities: List[float] = []
            for raw_answer, log_prob, correct_answer in zip(raw_answers, log_probs, correct_answers):
                print(raw_answer)
                answer = raw_answer[0].split(";")[0].lower()
                assert isinstance(correct_answer, str), \
                    f"String expected but got {correct_answer} of type {type(correct_answer)} (1)"
                utility = (answer == correct_answer)
                utilities.append(utility)
                single_loss = - log_prob * utility
                loss_list.append(single_loss)

            print("utilities:", utilities)
            mean_utility = np.mean(np.array(utilities))
            total_loss = torch.mean(torch.stack(loss_list))

            print("loss:", total_loss.item())
            optimizer.zero_grad()
            total_loss.backward()
            print("Grad:", self._swarm.connection_dist.edge_logits.grad)
            optimizer.step()

            print("edge_logits:", self._swarm.connection_dist.edge_logits)
            edge_probs = torch.sigmoid(self._swarm.connection_dist.edge_logits)
            print("edge_probs:", edge_probs)

            self.logger.log(metric_name="train_loss", value=total_loss.item(),
                            log_stdout=True, log_wandb=True)
            self.logger.log(metric_name="train_utility", value=mean_utility.item(),
                            log_stdout=True, log_wandb=True)

            self.logger.info("end of iteration")

        print("Done!")
        edge_probs = torch.sigmoid(self._swarm.connection_dist.edge_logits)
        return edge_probs


def main():
    dset = SyntheticEmotionDataset("data/synthetic_dialogues/v2/dialogues.json",
                                   "data/synthetic_dialogues/v2/scenarios.json")

    swarm = Swarm(["InsideOutCOT", "InsideOutCOT"], "gaia", model_name="gpt-4o-mini",
                  edge_optimize=True, )

    config = {
        "lr": 0.1,
        "num_iters": 30
    }
    logger = Logger(
        group="gptswarm_erc_debug",
        run_name="test_run2",
        tags=["debug"],
        config=config,
        use_wandb=True,
    )

    evaluator = Evaluator(
        swarm,
        dset,
        logger
    )
    evaluator.optimize_swarm(num_iters=config["num_iters"], lr=config["lr"])


if __name__ == "__main__":
    main()
