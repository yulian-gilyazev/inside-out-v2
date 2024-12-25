from enum import Enum


class Emotion(Enum):
    ANGER = "anger"
    DISGUST = "disgust"
    FEAR = "fear"
    HAPPINESS = "happiness"
    SADNESS = "sadness"

    @classmethod
    def from_str(cls, value):
        for name, member in cls.__members__.items():
            if member.value == value:
                return member
        raise ValueError(f'{value} is not a valid {cls.__name__}')


