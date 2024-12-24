import sys
import os
sys.path.append(os.path.join("libs", "GPTSwarm"))

from swarm.graph.swarm import Swarm


def main():
    swarm = Swarm(["IO", "IO", "IO"], "gaia")
    task = "What is the capital of Jordan?"
    inputs = {"task": task}
    answer = swarm.arun(inputs)
    print(answer)


if __name__ == "__main__":
    main()