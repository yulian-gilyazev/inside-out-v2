from src.schema.emotions import Emotion, EmpatheticDialoguesEmotion, EmpatheticDialoguesTruncatedEmotion,EmotionSet


def get_emotions_list_str(emotion_set: EmotionSet) -> str:
    if emotion_set == EmotionSet.EMPATHETIC_DIALOGUES:
        emotions_cls = EmpatheticDialoguesEmotion
    elif emotion_set == EmotionSet.TRUNCATED:
        emotions_cls = EmpatheticDialoguesTruncatedEmotion
    else:
        emotions_cls = Emotion
    emotions_list = [emotion.lower().capitalize() for emotion in emotions_cls.__members__.keys()]
    return emotions_list

def get_inside_out_emotinoal_prompt(emotion_set: EmotionSet) -> str:
    emotions_list = get_emotions_list_str(emotion_set)
    emotions_list_str = ", ".join(emotions_list)    
    n_emotions = len(emotions_list)
    if emotion_set == EmotionSet.EKMAN:
        emotions_list_str = f"use Ekman's classification into {n_emotions} main emotions - {emotions_list_str}"
    else:
        emotions_list_str = f"use classification into {n_emotions} emotions - {emotions_list_str}"

    return f"""You represent {{emotion}}. You should focus on this emotion in any query and act based on what emotion you are experiencing.
You need to assess emotion of the first (A) interlocutor in the dialogue and estimate your confidence.
Your answer should consist of an emotion and an assessment of the level of confidence in it in the range from 0 to 1.
To select emotions, {emotions_list_str}. 
Separate the emotion and the response using a semicolon.
Response example:
`Anger; 0.7`"""


def get_inside_out_emotinoal_prompt_predebate(emotion_set: EmotionSet) -> str:
    emotions_list = get_emotions_list_str(emotion_set)
    emotions_list_str = ", ".join(emotions_list)    
    n_emotions = len(emotions_list)
    if emotion_set == EmotionSet.EKMAN:
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


def get_inside_out_aggregator_prompt(emotion_set: EmotionSet) -> str:
    emotions_list = get_emotions_list_str(emotion_set)
    emotions_list_str = ", ".join(emotions_list)
    n_emotions = len(emotions_list)
    if emotion_set == EmotionSet.EKMAN:
        emotions_list_str = f"use Ekman's classification into {n_emotions} main emotions - {emotions_list_str}"
    else:
        emotions_list_str = f"use classification into {n_emotions} emotions - {emotions_list_str}"
    return f"""You have been given answers by several emotional agents, each of whom was interviewed to assess the emotional state of the first (A) interlocutor in the dialogue.
Each agent experiences their own emotions, which will be reflected in their name before they respond, and these emotions affect how they perceive the emotions of the first () in the conversation. You need to consider this information when solving the problem further.
You are also given the dialogue itself.
Your task is to aggregate the responses of these agents and give your own based on the dialogue and the responses of the agents.
Your answer should consist of an emotion and an assessment of the level of confidence in it in the range from 0 to 1.
To select emotions, {emotions_list_str}.
Separate the emotion and the response using a semicolon.
Response example:
`Anger; 0.7`
The same format is followed for agent responses.
"""


def get_emotions_generation_prompt(emotion_set: EmotionSet) -> str:
    emotions_list = get_emotions_list_str(emotion_set)
    emotions_list_str = ", ".join(emotions_list)
    n_emotions = len(emotions_list)

    if emotion_set == EmotionSet.EKMAN:
        emotions_list_str = f"{n_emotions} emotions from Ekman’s classification: {emotions_list_str}."
    else:
        emotions_list_str = f"{n_emotions} emotions: {emotions_list_str}."

    return f"""You are tasked with generating a set of emotional states to assist an agent-based dialogue evaluator in accurately identifying the first speaker’s underlying emotion in a conversation.

Your generated emotional states should guide the evaluator by offering emotionally plausible interpretations grounded in the dialogue context. The quality and usefulness of the agent system’s emotional recognition depend directly on the relevance and clarity of these emotions.
Be careful and choose emotional states from the perspective of which it would be easy to correctly assess the emotion of the interlocutor in the dialogue.

**Instructions:**

1. **Help the evaluator:**  Avoid introducing emotional states that can prevent the evaluator from correctly assessing the emotion of the interlocutor in the dialogue. And in opposite, choose emotional states that would be easy to correctly assess the emotion of the interlocutor in the given dialogue.

2. **Emotional Diversity:** Depending on the dialogue’s complexity, provide between **2 and 8 distinct emotional states**. Use nuanced or compound emotions when appropriate (e.g., <EMOTION>Anger and Betrayal</EMOTION>), but avoid redundancy and repetition.

3. **Formatting Rules (Strict):**
   - Each emotion must be wrapped in `<EMOTION>` tags. 
     *Example:* `<EMOTION>Frustration</EMOTION>`
   - Combinations of two emotions are allowed using “and”.  
     *Example:* `<EMOTION>Fear and Disgust</EMOTION>`
   - Emotions must be selected only from this predefined list: {emotions_list_str}.
   - Do **not** duplicate any emotion or combination within a response.

4. **Impact on Evaluator:** Your emotional choices should help disambiguate the first speaker’s emotional state, enabling the agent system to synthesize accurate insights when integrating multiple agent perspectives.

5. **Provide your reasoning:** Provide your reasoning for the emotional states you chose before generating the list of emotional states. Do not exceed 2-6 sentences.

**Example Output:**

```
<REASONING>
...
</REASONING>
<EMOTION>Frustration</EMOTION> <EMOTION>Sadness and Frustration</EMOTION> <EMOTION>Disgust</EMOTION>
```

**Important:** The generated emotional states will directly influence the evaluator’s interpretations. Aim for emotional labels that are both **plausible** and **diagnostically useful**.
"""

def get_emotional_agent_debate_prompt(emotion_set: EmotionSet) -> str:
    emotions_list = get_emotions_list_str(emotion_set)
    emotions_list_str = ", ".join(emotions_list)    
    n_emotions = len(emotions_list)
    if emotion_set == EmotionSet.EKMAN:
        emotions_list_str = f"use Ekman's classification into {n_emotions} main emotions - {emotions_list_str}"
    else:
        emotions_list_str = f"use classification into {n_emotions} emotions - {emotions_list_str}"

    return f"""You feel {{emotion}}. You should focus on this emotion in any query and act based on what emotion you are experiencing.
You need to assess emotion of the first (A) interlocutor in the dialogue and estimate your confidence.
You will be provided with the following supplementary information to support your evaluation:  
- Your own previous assessment from the last round.  
- The current assessments made by other emotional agents, each analyzing Speaker A’s emotional state through the lens of a different emotion.

All previous assessments will be presented as a single list, which includes your own prior evaluation.

Use this context to revise and improve your current prediction, taking into account both your past perspective and the varying emotional viewpoints of other agents in the system.
The ultimate goal of this round is to provide the most accurate assessment of the emotion of the interlocutor.

Your answer should consist of an emotion and an assessment of the level of confidence in it in the range from 0 to 1.
To select emotions, {emotions_list_str}. 
Separate the emotion and the response using a semicolon.
Response example:
`Anger; 0.7`
"""


def get_system_prompt() -> str:
    return """You are a highly advanced language model.
Carefully heed the user's instructions."""