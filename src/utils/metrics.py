from src.llm_client import LLMCausalProbabilityClient
from typing import List, Dict
import numpy as np

def log_prob_of_completion(llm_causal_probability_client: LLMCausalProbabilityClient, prompt_messages: List[Dict[str, str]], completion: str) -> float:
    """
    Calculate the log probability of a completion.
    """
    return llm_causal_probability_client.log_prob_of_text(prompt_messages, completion)["total_log_prob"]


def position_weighted_log_prob_of_completion(log_probabilities: np.ndarray, decay_factor=0.98):
    """
    Calculate the position-weighted log probability of a completion.
    """
    
    weighted_log_probs = []
    for i, log_prob in enumerate(log_probabilities):
        weight = decay_factor ** i
        weighted_log_probs.append(log_prob * weight)
    
    total_weight = sum(decay_factor ** i for i in range(log_probabilities.shape[0]))
    weighted_avg = sum(weighted_log_probs) / total_weight
    
    return weighted_avg


def dpo_loss(total_log_probabilities_anchor: np.ndarray, total_log_probabilities_negative: np.ndarray):
    """
    Calculate the DPO loss between two sets of log probabilities.
    """
    return -total_log_probabilities_anchor + total_log_probabilities_negative

def accuracy_at_k(total_log_probabilities_anchor: np.ndarray, total_log_probabilities_negative_samples: np.ndarray):
    """
    Calculate the accuracy at k between two sets of log probabilities.
    """
    all_probabilities = np.concatenate([[total_log_probabilities_anchor], total_log_probabilities_negative_samples])
    
    sorted_indices = np.argsort(-all_probabilities)
    
    anchor_position = np.where(sorted_indices == 0)[0][0]
    return anchor_position

def acc_at_k_position_weighted(log_probabilities_anchor: np.ndarray, log_probabilities_negative: np.ndarray, k: int):
    """
    Calculate the accuracy at k between two sets of log probabilities.
    """
    return np.sum(log_probabilities_anchor[:k]) > np.sum(log_probabilities_negative[:k])


