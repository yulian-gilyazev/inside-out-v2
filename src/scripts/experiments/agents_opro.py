import json
import tempfile
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Literal, Optional, Tuple

from tqdm.auto import tqdm

from src.agent import Pipeline
from src.agent.registry import PipelineAgentConfig
from src.llm_client import LLMClient
from src.models.opro import OPRO
from src.scripts.experiments.utils import evaluate_pipeline
from src.utils.data import EmotionDataset
from src.utils.logger import Logger


@dataclass
class OptimizePipelineConfig:
    n_steps: int = 4
    opro_memory_strategy: Literal["last", "all"] = "all"
    num_workers: int = 4
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def optimize_pipeline(llm_client: LLMClient, 
                      llm_prompt_searcher: LLMClient,
                      logger: Logger,
                      optimize_config: OptimizePipelineConfig,
                      train_dset: EmotionDataset,
                      test_dset: EmotionDataset,
                      opro_metaprompt: str,
                      opro_task_prompt: str,
                      opro_prompt_tokens: Tuple[str, str],
                      prompts: Dict[str, str], 
                      cfg_from_prompts_fn: Callable[[Dict[str, str]], PipelineAgentConfig],
                      check_fn: Optional[Callable[[str], bool]] = None,
                      ):
    """
    Optimize the pipeline using OPRO.
    """

    if check_fn is None:
        check_fn = lambda x: True

    optimizer = OPRO(llm_prompt_searcher,
                    "accuracy", 
                    opro_metaprompt,
                    opro_task_prompt, 
                    check_fn=check_fn,
                    prompt_tokens=opro_prompt_tokens,
                    memory_strategy=optimize_config.opro_memory_strategy,
                    logger=logger)
    
    pipeline_prompts_json = json.dumps(prompts)

    for step in tqdm(range(optimize_config.n_steps)):
        pipeline_prompts = json.loads(pipeline_prompts_json)
        pipeline_cfg = cfg_from_prompts_fn(**pipeline_prompts)
        pipeline = Pipeline(pipeline_cfg, llm_client)
        train_acc = evaluate_pipeline(pipeline, pipeline_cfg, train_dset)
        logger.log(
            metric_name="train_accuracy",
            value=train_acc,
            log_stdout=True,
            log_wandb=True,
        )
        test_acc = evaluate_pipeline(pipeline, pipeline_cfg, test_dset)
        logger.log(
            metric_name="test_accuracy",
            value=test_acc,
            log_stdout=True,
            log_wandb=True,
        )
        if step == 0:
            optimizer.initialize(pipeline_prompts_json, reward=train_acc)
            new_prompt = optimizer.step()
            pipeline_prompts_json = new_prompt
        else:
            optimizer.send_reward(train_acc)
            new_prompt = optimizer.step()
            pipeline_prompts_json = new_prompt
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            cfg_path = f.name
            f.write(pipeline_prompts_json.encode("utf-8"))
            artifact = logger.wandb.Artifact(name=f"config_{step}", type="dataset")
            artifact.add_file(cfg_path)
            logger.run.log_artifact(artifact)
            logger.info(f"Config {step} saved")
            logger.info(f"Config: \n{pipeline_prompts_json}")
    return pipeline