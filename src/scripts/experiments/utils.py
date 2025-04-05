import numpy as np
from tqdm import tqdm
from src.agent import Pipeline, PipelineAgentConfig, AgentContext
from src.utils.data import SyntheticEmotionDataset

from concurrent.futures import ThreadPoolExecutor
from functools import partial


def accuracy(gt, pred):
    mask = [u == v for u, v in zip(gt, pred)]
    return np.array(mask).mean()


def run_pipeline(pipeline: Pipeline, pipeline_cfg: PipelineAgentConfig, dset: SyntheticEmotionDataset, num_workers: int = 4) -> float:
    """
    Run the pipeline on the dataset.
    """
    predictions = []
    gt = []
    
    def process_item(item, pipeline, pipeline_cfg):
        context = AgentContext(data={"input": item.format_dialogue()})
        context = pipeline.process(context)
        predicted = context.get_value(pipeline_cfg.output_id)
        return predicted, item.emotion.value.lower()
    
    process_func = partial(process_item, pipeline=pipeline, pipeline_cfg=pipeline_cfg)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_func, dset), total=len(dset)))
    
    predictions = [res[0] for res in results]
    gt = [res[1] for res in results]

    return predictions, gt


def evaluate_pipeline(pipeline: Pipeline, pipeline_cfg: PipelineAgentConfig, dset: SyntheticEmotionDataset, num_workers: int = 4) -> float:
    """
    Evaluate the pipeline on the dataset.
    """
    predictions, gt = run_pipeline(pipeline, pipeline_cfg, dset, num_workers)
    predictions = [pred.split(";")[0].lower() for pred in predictions]
    return accuracy(gt, predictions)
