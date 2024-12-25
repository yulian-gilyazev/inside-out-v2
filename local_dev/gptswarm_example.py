import sys
import os
sys.path.append(os.path.join("libs", "GPTSwarm"))

from swarm.graph.swarm import Swarm

from src.config import OPENAI_API_KEY


def main():

    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    swarm = Swarm(["IO", "IO", "IO"], "gaia", model_name="openai/gpt-4o-mini")
    task = "What is the capital of Jordan?"
    inputs = {"task": task}
    answer = swarm.run(inputs)
    print(answer)


if __name__ == "__main__":
    main()