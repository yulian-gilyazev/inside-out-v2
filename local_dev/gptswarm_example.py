import sys
import os
from loguru import logger
sys.path.append(os.path.join("libs", "GPTSwarm"))

from swarm.graph.swarm import Swarm

from src.config import VSEGPT_API_KEY

def main():

    os.environ["OPENAI_API_KEY"] = VSEGPT_API_KEY
    logger.info("Initializing swarm")

    swarm = Swarm(["IO", "IO", "IO"], "gaia", model_name="gpt-4o-mini")
    task = "What is the capital of Russia?"
    inputs = {"task": task}
    answer = swarm.run(inputs)

    logger.info(f"Answer: {answer}")


if __name__ == "__main__":
    main()