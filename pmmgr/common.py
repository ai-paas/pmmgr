import logging.config
from pydantic_settings import BaseSettings
from enum import StrEnum
import logging
from typing import List


class Settings(BaseSettings):
    DB_PATH: str = "data/db/pmmgr.db"
    TEST_DB_PATH: str = "tmp/test.db"
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"


class TaskType(StrEnum):
    SEPARATOR = ","
    TEXT_GENERATION = "Text Generation"
    TEXT_CLASSIFICATION = "Text Classification"
    QUESTION_ANSWERING = "Question Answering"
    SUMMARIZATION = "Summarization"
    TRANSLATION = "Translation"
    NAMED_ENTITY_RECOGNITION = "Named Entity Recognition"
    SENTIMENT_ANALYSIS = "Sentiment Analysis"

    @classmethod
    def values(cls):
        return [task_type.value for task_type in cls]

    @classmethod
    def str_to_list(cls, task_types: str):
        return task_types.split(cls.SEPARATOR)

    @classmethod
    def list_to_str(cls, task_types: List[str]):
        return cls.SEPARATOR.join(task_types)

class IOType(StrEnum):
    TEXT = "Text"
    IMAGE = "Image"
    AUDIO = "Audio"
    VIDEO = "Video"


class QuantType(StrEnum):
    FP16 = "FP16"
    BF16 = "BF16"
    Q8 = "Q8"
    Q5 = "Q5"
    Q4 = "Q4"
    
    @classmethod
    def values(cls):
        return [quant_type.value for quant_type in cls]
    

settings = Settings()
logging.config.fileConfig('config/logging.conf')
logger = logging.getLogger('pmmgr')
