from src.schema.emotions import Emotion, EmpatheticDialoguesEmotion


def get_emotions_list_str(is_extended: bool) -> str:
    if is_extended:
        emotions_cls = EmpatheticDialoguesEmotion
    else:
        emotions_cls = Emotion
    emotions_list = [emotion.lower().capitalize() for emotion in emotions_cls.__members__.keys()]
    return emotions_list

def get_inside_out_emotinoal_prompt(is_extended: bool) -> str:
    emotions_list = get_emotions_list_str(is_extended)
    emotions_list_str = ", ".join(emotions_list)    
    n_emotions = len(emotions_list)
    if not is_extended:
        emotions_list_str = f"use Ekman's classification into {n_emotions} main emotions - {emotions_list_str}"
    else:
        emotions_list_str = f"use classification into {n_emotions} emotions - {emotions_list_str}"

    return f"""You feel {{emotion}}. Act based on what emotion you are experiencing.
You need to assess emotion of the first (A) interlocutor in the dialogue and estimate your confidence.
Your answer should consist of an emotion and an assessment of the level of confidence in it in the range from 0 to 1.
To select emotions, {emotions_list_str}. 
Separate the emotion and the response using a semicolon.
Response example:
`Anger; 0.7`"""
    
def get_inside_out_aggregator_prompt(is_extended: bool) -> str:
    emotions_list = get_emotions_list_str(is_extended)
    emotions_list_str = ", ".join(emotions_list)
    n_emotions = len(emotions_list)
    if not is_extended:
        emotions_list_str = f"use Ekman's classification into {n_emotions} main emotions - {emotions_list_str}"
    else:
        emotions_list_str = f"use classification into {n_emotions} emotions - {emotions_list_str}"
    return f"""You have been given answers by several emotional agents, each of whom was interviewed to assess the emotional state of the first (A) interlocutor in the dialogue.
You are also given the dialogue itself.
Your task is to aggregate the responses of these agents and give your own based on the dialogue and the responses of the agents.
Your answer should consist of an emotion and an assessment of the level of confidence in it in the range from 0 to 1.
To select emotions, {emotions_list_str}.
Separate the emotion and the response using a semicolon.
Response example:
`Anger; 0.7`
The same format is followed for agent responses.
"""


def get_emotions_generation_prompt(is_extended: bool) -> str:
    emotions_list = get_emotions_list_str(is_extended)
    emotions_list_str = ", ".join(emotions_list)
    n_emotions = len(emotions_list)

    if is_extended:
        emotions_list_str = f"{n_emotions} emotions from Ekman’s classification: {emotions_list_str}."
    else:
        emotions_list_str = f"{n_emotions} emotions: {emotions_list_str}."

    return f"""Your assignment is to propose a range of emotional states meant for a dialogue evaluator whose objective is to determine the first speaker’s emotion in the dialogue. 

By providing both the dialogue and the emotional states you generate, you empower the evaluator to more accurately identify the target speaker’s emotion.

Key Considerations:
- Your selection of emotional states directly affects the evaluator’s accuracy. Some emotions will help clarify the first speaker’s emotion, while others may obscure it.
- Make sure your suggestions are grounded in the dialogue context. Conflicting dialogues rarely involve mutual happiness, so avoid adding emotions that create unnecessary confusion.
- Ensure the emotional states are credible and conducive to facilitating accurate recognition when different agent perspectives are combined.

Format Requirements:
• Place each emotion inside <EMOTION> tags, for example, <EMOTION>Anger</EMOTION>.
• Only use the {emotions_list_str}
• You can create combinations of two emotions like <EMOTION>Sadness and Disgust</EMOTION>.
• Do not repeat the same emotion or combination in different tags.
• Provide two to five unique emotional states, depending on the complexity of the dialogue.
• Example Output: <EMOTION>Anger</EMOTION> <EMOTION>Fear and Anger</EMOTION> <EMOTION>Sadness</EMOTION>

Remember: your chosen emotions will heavily influence the evaluator’s performance."""


def get_emotional_agent_debate_prompt() -> str:
    return """You will also be given the responses from other emotional agents and your own response from the previous round of debate. This information will help you give your answer more confidently.
Using the solutions from other emotional agents (each agent has the same task as you, but feels different emotions) and your own response from the previous round of debate as additional information, give a response. 
Emotional agents responses:\n{prev_round_key}\n\n\n Dialogue:\n{input}."""

def get_system_prompt() -> str:
    return """You are a highly advanced language model.
Carefully heed the user's instructions."""