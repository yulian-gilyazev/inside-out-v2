import wandb
from loguru import logger
from typing import Any, Dict, List

from src.config import WANDB_PROJECT_NAME


class Logger:
    def __init__(self, group: str = None, run_name: str = None, config: Dict = None, notes: str = None,
                 tags: List[str] = None, use_wandb: bool = True):
        self.use_wandb = use_wandb
        if use_wandb:
            self.run = wandb.init(project=WANDB_PROJECT_NAME,
                                  group=group, name=run_name,
                                  notes=notes, tags=tags, config=config)

    @staticmethod
    def info(message):
        logger.info(message)

    @staticmethod
    def warning(message):
        logger.warning(message)

    @staticmethod
    def error(message):
        logger.error(message)

    def log_wandb(self, metric_name, value):
        if self.use_wandb:
            self.run.log({metric_name: value})
        else:
            logger.error("Wandb logging is not enabled")
            logger.info(f"{metric_name}: {value}")

    def log(self, metric_name: str, value: Any, log_stdout: bool = False, log_wandb: bool = False):
        if not log_wandb and not log_stdout:
            logger.error("Both wandb ang loguru logging are not enabled")
            return
        if log_wandb:
            if self.use_wandb:
                self.run.log({metric_name: value}, commit=False)
            else:
                logger.error("Wandb logging is not enabled")
        if log_stdout:
            logger.info(f"{metric_name}: {value}")
