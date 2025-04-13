from enum import Enum


class EmotionSet(Enum):
    EKMAN = "ekman"
    EMPATHETIC_DIALOGUES = "emp_dialogues"
    TRUNCATED = "truncated"
    

    @classmethod
    def from_str(cls, value):
        for name, member in cls.__members__.items():
            if member.value == value:
                return member
        raise ValueError(f'{value} is not a valid {cls.__name__}')

class BaseEmotion(Enum):
    @classmethod
    def from_str(cls, value):
        for name, member in cls.__members__.items():
            if member.value == value:
                return member
        raise ValueError(f'{value} is not a valid {cls.__name__}')


class Emotion(BaseEmotion):
    ANGER = "anger"
    DISGUST = "disgust"
    FEAR = "fear"
    HAPPINESS = "happiness"
    SADNESS = "sadness"


class EmpatheticDialoguesEmotion(BaseEmotion):
    SENTIMENTALITY = "sentimentality"
    FEAR = "fear"
    PRIDE = "pride"
    FAITHFULNESS = "faithfulness"
    TERROR = "terror"
    JOY = "joy"
    ANGER = "anger"
    SADNESS = "sadness"
    JEALOUSY = "jealousy"
    GRATEFULNESS = "gratefulness"
    PREPAREDNESS = "preparedness"
    EMBARRASSMENT = "embarrassment"
    EXCITEMENT = "excitement"
    ANNOYANCE = "annoyance"
    LONELINESS = "loneliness"
    SHAME = "shame"
    GUILT = "guilt"
    SURPRISE = "surprise"
    NOSTALGIA = "nostalgia"
    CONFIDENCE = "confidence"
    FURY = "fury"
    DISAPPOINTMENT = "disappointment"
    CARING = "caring"
    TRUSTING = "trusting"
    DISGUST = "disgust"
    ANTICIPATION = "anticipation"
    ANXIOUSNESS = "anxiousness"
    HOPEFULNESS = "hopefulness"
    CONTENTMENT = "contentment"
    IMPRESSION = "impression"
    APPREHENSION = "apprehension"
    DEVASATION = "devastation"
    

    @classmethod
    def emotion_to_empathy_dialogues_emotion_mapping(self):
        return {
            "sentimentality": "sentimental",
            "fear": "afraid",
            "pride": "proud",
            "faithfulness": "faithful",
            "terror": "terrified",
            "joy": "joyful",
            "anger": "angry",
            "sadness": "sad",
            "jealousy": "jealous",
            "gratefulness": "grateful",
            "preparedness": "prepared",
            "embarrassment": "embarrassed",
            "excitement": "excited",
            "annoyance": "annoyed",
            "loneliness": "lonely",
            "shame": "ashamed",
            "guilt": "guilty",
            "surprise": "surprised",
            "nostalgia": "nostalgic",
            "confidence": "confident",
            "fury": "furious",
            "disappointment": "disappointed",
            "caring": "caring",
            "trusting": "trusting",
            "disgust": "disgusted",
            "anticipation": "anticipating",
            "anxiousness": "anxious",
            "hopefulness": "hopeful",
            "contentment": "content",
            "impression": "impressed",
            "apprehension": "apprehensive",
            "devastation": "devastated",
        }
    
    @classmethod
    def empathy_dialogues_emotion_to_emotion(cls, empathy_dialogues_emotion: str) -> BaseEmotion:
        mapping = cls.emotion_to_empathy_dialogues_emotion_mapping()
        reverse_mapping = {v: k for k, v in mapping.items()}
        return EmpatheticDialoguesEmotion.from_str(reverse_mapping[empathy_dialogues_emotion])

    @classmethod
    def emotion_to_empathy_dialogues_emotion(cls, emotion: BaseEmotion) -> str:
        mapping = cls.emotion_to_empathy_dialogues_emotion_mapping()
        return mapping[emotion.value]



class EmpatheticDialoguesTruncatedEmotion(BaseEmotion):
    SENTIMENTALITY = "sentimentality"
    FEAR = "fear"
    HAPPINESS = "happiness"
    PRIDE = "pride"
    ANGER = "anger"
    SADNESS = "sadness"
    JEALOUSY = "jealousy"
    GRATEFULNESS = "gratefulness"
    EMBARRASSMENT = "embarrassment"
    GUILT = "guilt"
    SURPRISE = "surprise"
    CONFIDENCE = "confidence"
    CARING = "caring"
    TRUSTING = "trusting"
    DISGUST = "disgust"
    ANXIOUSNESS = "anxiousness"
    HOPEFULNESS = "hopefulness"

    @classmethod
    def empathy_dialogues_emotion_to_emotion_mapping(self):
        return {
            "sentimental": "sentimentality", 
            "afraid": "fear",
            "proud": "pride",
            "faithful": "trusting",
            "terrified": "fear",
            "joyful": "happiness",
            "angry": "anger",
            "sad": "sadness",
            "jealous": "jealousy",
            "grateful": "gratefulness",
            "prepared": "anxiousness",
            "embarrassed": "embarrassment",
            "excited": "happiness",
            "annoyed": "anger",
            "lonely": "sadness",
            "ashamed": "embarrassment",
            "surprised": "surprise",
            "nostalgic": "sentimentality",
            "confident": "confidence",
            "furious": "anger",
            "disappointed": "sadness",
            "caring": "caring",
            "guilty": "guilt",
            "trusting": "trusting",
            "disgusted": "disgust",
            "anticipating": "anxiousness",
            "anxious": "anxiousness",
            "hopeful": "hopefulness",
            "content": "happiness",
            "impressed": "surprise",
            "apprehensive": "anxiousness",
            "devastated": "sadness",
        }
    
    @classmethod
    def empathy_dialogues_emotion_to_emotion(cls, empathy_dialogues_emotion: str) -> BaseEmotion:
        mapping = cls.empathy_dialogues_emotion_to_emotion_mapping()
        return EmpatheticDialoguesTruncatedEmotion.from_str(mapping[empathy_dialogues_emotion])

    @classmethod
    def emotion_to_empathy_dialogues_emotion(cls, emotion: BaseEmotion) -> str:
        mapping = cls.emotion_to_empathy_dialogues_emotion_mapping()
        reverse_mapping = {v: k for k, v in mapping.items()}
        return reverse_mapping[emotion.value]
