import json
import logging.config
import os
import pickle
import unittest
import warnings

from pydantic_settings import BaseSettings
from enum import StrEnum, auto
import logging
from typing import List


class Settings(BaseSettings):
    DB_PATH: str = "/trunk/pmmgr/data/db/pmmgr.db"
    TEST_DB_PATH: str = "/trunk/pmmgr/data/db/test.db"
    LOGGING_CONFIG_PATH: str = "/trunk/pmmgr/configs/logging.conf"
    PM_INFO_CSV: str = "/trunk/pmmgr/data/pminfo.csv"
    TEST_PM_INFO_CSV: str = "/trunk/pmmgr/data/test_pminfo.csv"
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
    NONE = "-"
    FP16 = "FP16"
    BF16 = "BF16"
    Q8 = "Q8"
    Q5 = "Q5"
    Q4 = "Q4"
    
    @classmethod
    def values(cls):
        return [quant_type.value for quant_type in cls]
    
class ModelType(StrEnum):
    OLLAMA = "ollama"
    LLAMA_CPP = "llamacpp"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    VLLM = "vllm"
    GOOGLE = "google"

class LicenseType(StrEnum):
    APACHE20 = "Apache-2.0"
    MIT = "MIT"
    GPL = "GPL"
    BSD = "BSD"
    UNLICENSED = "UNLICENSED"

class EvalMetric(StrEnum):
    ACCURACY = "Accuracy"
    F1 = "F1"
    RECALL = "Recall"
    PRECISION = "Precision"
    ROC_AUC = "ROC-AUC"
    MMLU = "MMLU"
    MMLU_REDUX = "MMLU-Redux"
    GPQA_DIAMOND = "GPQA-Diamond"
    AVG_MR_GPQA = "Average of MMLU-Redux, GPQA-Diamond"

class MessageCode(StrEnum):
    NO_AVAILABLE_CHAT_MODEL = auto()
    DB_ERROR = auto()
    UNKNOWN_ERROR = auto()

    FN_MSG_MAP = '../configs/msg.json'

    @classmethod
    def get_message(cls, msg_code):
        if not hasattr(cls, 'msg_map'):
            cls.msg_map = cls.load_msg_map()
        return cls.msg_map.get(msg_code, 'Unknown message code')

    @classmethod
    def load_msg_map(cls):
        return FileUtils.json_load(cls.FN_MSG_MAP)

class MessageCodeException(Exception):
    def __init__(self, msg_code, sub_msg=None):
        super().__init__(f"{msg_code}: {MessageCode.get_message(msg_code)}{'-' + sub_msg if sub_msg else ''}")


class FileUtils(object):
    @staticmethod
    def json_load(fn, decoder=None):
        data = None
        with open(fn, 'r') as f:
            data = json.load(f, cls=decoder)
        return data

    @staticmethod
    def json_dump(data, fn, ensure_ascii=False, encoder=None):
        with open(fn, 'w') as f:
            json.dump(data, f, ensure_ascii=ensure_ascii, cls=encoder)

    @staticmethod
    def pkl_load(fn):
        data = None
        with open(fn, 'rb') as f:
            data = pickle.load(f)
        return data

    @staticmethod
    def pkl_dump(data, fn):
        with open(fn, 'wb') as f:
            pickle.dump(data, f)

    # @staticmethod
    # def get_last_version(dirpath, basename):
    #     fns = glob.glob(fn_pattern)
    #     if len(fns) == 0:
    #         return None
    #     the = np.argmax([StrUtils.search_digit(fn[len(basename):], default=0, last=True) for fn in fns])
    #     return sorted(fns)[-1]

    @classmethod
    def str_load(cls, fn):
        with open(fn, 'r') as f:
            return f.read()

    @classmethod
    def append_fn_suffix(cls, file_path, fn_suffix):
        tokens = os.path.splitext(file_path)
        return f"{tokens[0]}{fn_suffix}{tokens[1]}"

settings = Settings()

class BaseTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        warnings.filterwarnings('ignore')
        logging.config.fileConfig(settings.LOGGING_CONFIG_PATH)

