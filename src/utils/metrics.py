from src.llm_client import LLMCausalProbabilityClient
from typing import List, Dict


def log_prob_of_text(llm_causal_probability_client: LLMCausalProbabilityClient, prompt_messages: List[Dict[str, str]], completion: str) -> float:
    """
    Calculate the log probability of a completion.
    """
    return llm_causal_probability_client.log_prob_of_text(prompt_messages, completion)["total_log_prob"]


def position_weighted_log_prob(llm_causal_probability_client, prompt_messages, completion, decay_factor=0.98):
    """
    Calculate the position-weighted log probability of a completion.
    """
    completions_prob = llm_causal_probability_client.log_prob_of_text(prompt_messages, completion)
    
    weighted_log_probs = []
    for i, log_prob in enumerate(completions_prob["log_probabilities"]):
        weight = decay_factor ** i
        weighted_log_probs.append(log_prob * weight)
    
    total_weight = sum(decay_factor ** i for i in range(len(completions_prob["log_probabilities"])))
    weighted_avg = sum(weighted_log_probs) / total_weight
    
    return weighted_avg
