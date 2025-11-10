"""
This module is for registering chat models.
"""
import logging
import os
import unittest
from collections import OrderedDict
from time import sleep

from typing import List, Any, Dict
from unittest import TestCase
from unittest.mock import patch

from langchain_ollama import ChatOllama
from langchain_community.llms import LlamaCpp
from pydantic import Field, validator, field_validator, root_validator, model_validator
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

import db
from pmmgr.common import MessageCodeException, MessageCode, ModelType, settings, BaseTest

###
# Logger
logger = logging.getLogger('pmmgr')


class MultiChatModel(BaseChatModel):
    class ModelWrapper(BaseChatModel):
        minfo: dict = Field(default_factory=dict)
        cm: BaseChatModel = None
        ready: bool = True

        @property
        def _llm_type(self):
            return self.cm._llm_type

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            try:
                self.ready = False
                result = self.cm._generate(messages, stop, run_manager, **kwargs)
                return result
            except Exception as e:
                logger.error(f'Error in _generate of {self}: {e}')
                raise e
            finally:
                self.ready = True

        def _stream(self, messages, stop=None, run_manager=None, **kwargs):
            try:
                self.ready = False
                self.cm._stream(messages, stop, run_manager, **kwargs)
            except Exception as e:
                logger.error(f'Error in _stream of {self}: {e}')
                raise e
            finally:
                self.ready = True


    MAX_RETRY: int = 10
    DELAY: int = 2

    cms: List[ModelWrapper] = Field(default_factory=list)

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f'MultiChatModel(llm_type={self._llm_type}, n_models={self.n_models}), id: {id(self)}'

    @property
    def _llm_type(self):
        return self.cms[0]._llm_type

    @property
    def minfo(self):
        return self.cms[0].minfo

    @property
    def n_models(self):
        return len(self.cms)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        cm = self.get_ready_chat_model()
        logger.debug(f'>>>Call _generate of {self}')
        return cm._generate(messages, stop, run_manager, **kwargs)

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        cm = self.get_ready_chat_model()
        logger.debug(f'>>>Call _stream of {self}')
        cm._stream(messages, stop, run_manager, **kwargs)

    def get_ready_chat_model(self):
        retry = 0
        while retry < self.MAX_RETRY:
            for i, cm in enumerate(self.cms):
                if cm.ready:
                    logger.info(f'Found {i}th ready chat model: {cm}')
                    return cm
            sleep(self.DELAY)
            retry += 1
            logger.info(f'No available chat model. Retry {retry}/{self.MAX_RETRY}')
        raise MessageCodeException(MessageCode.NO_AVAILABLE_CHAT_MODEL)


    @model_validator(mode='before')
    @classmethod
    def _wrap_chat_models(cls, data: Dict) -> Dict:
        minfo = data.pop('minfo', {})
        cms = data.get('cms', [])

        wrapped = []
        for model in cms:
            if isinstance(model, BaseChatModel) and not isinstance(model, cls.ModelWrapper):
                wrapped.append(cls.ModelWrapper(minfo=minfo, cm=model))

        data['cms'] = wrapped
        return data


class ChatModelPool(object):
    # DEFAULT_GOOGLE_SAFETY_SETTINGS = {
    #     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    #     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    #     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    #     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
    # }

    class Callback(object):
        def on_cm_updated(self, owner, id_or_path=None, cm=None):
            pass

    def __init__(self):
        self.cm_map = OrderedDict()
        self.callbacks = []
        self.init_cm_map()

    def init_cm_map(self, id_or_paths=None):
        self.cm_map.clear()

        pm_infos = db.list_pretrained_model_info()
        logger.info(f"Loaded {len(pm_infos)} pretrained model info")
        for pm_info in pm_infos:
            if pm_info.n_srv_instances > 0 and (id_or_paths is None or pm_info.id_or_path in id_or_paths):
                cm = self.create_multi_chat_model(pm_info=pm_info)
                self.cm_map[pm_info.id_or_path] = cm
                logger.info(f"Loaded the multi-chat model with {cm.n_models} {pm_info.mtype}(s) for {pm_info.id_or_path}")
                self.fire_cm_updated(self, pm_info.id_or_path, cm)

    # Callbacks
    def add_callback(self, cb):
        if cb not in self.callbacks:
            self.callbacks.append(cb)

    def remove_callback(self, cb):
        self.callbacks.remove(cb)

    def clear_callbacks(self):
        self.callbacks.clear()

    def fire_cm_updated(self, owner, id_or_path=None, cm=None):
        for cb in self.callbacks:
            cb.on_cm_updated(owner, id_or_path, cm)

    def get_chat_model(self, id_or_path):
        # if key not in self.cm_map:
        #     cm = self.create_chat_model(key)
        #     self.cm_map[key] = cm
        #     logger.info(f'Created multi chat model with {self.cm_map[key].n_models} source chat models for {key}')
        return self.cm_map[id_or_path]

    @classmethod
    def create_multi_chat_model(cls, pm_info=None):
        logger.info(f"Creating multi-chat model with {pm_info.n_srv_instances} instances for {pm_info.id_or_path}")
        return MultiChatModel(minfo=pm_info.to_dict(),
                              cms=[
                                  cls.create_chat_model(pm_info=pm_info) for _ in range(pm_info.n_srv_instances)
                              ])

    @classmethod
    def create_chat_model(cls, pm_info=None):
        cm_type =pm_info.mtype
        if cm_type == ModelType.OLLAMA:
            return ChatOllama(model=pm_info.id_or_path, **pm_info.config)
        elif cm_type == ModelType.LLAMA_CPP:
            return LlamaCpp(model_path=pm_info.id_or_path, **pm_info.config)
        else:
            raise NotImplementedError(f'{cm_type} chat model is not supported.')

    @classmethod
    def get_instance(cls):
        if not hasattr(cls, "_instance"):
            cls._instance = cls()
            logger.info(f'Created ChatModelPool instance')
        return cls._instance


class ChatModelPoolTest(BaseTest):

    def setUp(self):
        super().setUp()

        if os.path.exists(settings.TEST_DB_PATH):
            os.remove(settings.TEST_DB_PATH)
        db.create_pretrained_model_info_table(settings.TEST_DB_PATH)
        db.init_pretrained_model_info(settings.TEST_DB_PATH, csv_file=settings.TEST_PM_INFO_CSV)
        self.pm_info_map = OrderedDict()
        for pm_info in db.list_pretrained_model_info(settings.TEST_DB_PATH):
            self.pm_info_map[pm_info.id_or_path] = pm_info

    def tearDown(self):
        if os.path.exists(settings.TEST_DB_PATH):
            os.remove(settings.TEST_DB_PATH)

    def assert_mcm(self, mcm, cm_type=None, n_models=None):
        self.assertTrue(mcm.n_models == n_models)
        for cm in mcm.cms:
            self.assertIsNotNone(cm)
            self.assertTrue(isinstance(cm, MultiChatModel.ModelWrapper))
            self.assertTrue(cm.ready)
            if cm_type == ModelType.OLLAMA:
                self.assertTrue(isinstance(cm.cm, ChatOllama))
            if cm_type == ModelType.LLAMA_CPP:
                self.assertTrue(isinstance(cm.cm, LlamaCpp))

    def test_create_multi_chat_model(self):
        for pm_info in self.pm_info_map.values():
            mcm = ChatModelPool.create_multi_chat_model(pm_info=pm_info)
            self.assert_mcm(mcm, cm_type=pm_info.mtype, n_models=pm_info.n_srv_instances)

    @property
    def total_instances(self):
        return sum(pm_info.n_srv_instances for pm_info in self.pm_info_map.values())

    @property
    def net_n_instances(self):
        return sum(min(1, pm_info.n_srv_instances) for pm_info in self.pm_info_map.values())

    def test_init_cm_map(self):
        cm_pool = ChatModelPool.get_instance()
        cm_pool.init_cm_map(id_or_paths=self.pm_info_map.keys())
        self.assertTrue(len(cm_pool.cm_map) == self.net_n_instances)
        for id_or_path, mcm in cm_pool.cm_map.items():
            pm_info = self.pm_info_map[id_or_path]
            self.assert_mcm(mcm, cm_type=pm_info.mtype, n_models=pm_info.n_srv_instances)

    def test_get_instance(self):
        cm_pool = ChatModelPool.get_instance()
        self.assertIsNotNone(cm_pool)
        self.assertEqual(id(cm_pool), id(ChatModelPool.get_instance()))


class MultiChatModelTest(BaseTest):
    def setUp(self):
        super().setUp()
        self.cm_pool = ChatModelPool.get_instance()

    def test_invoke(self):
        for id_or_path in self.cm_pool.cm_map.keys():
            mcm = self.cm_pool.get_chat_model(id_or_path)
            self.assertIsNotNone(mcm)
            self.assertTrue(isinstance(mcm, MultiChatModel))
            result = mcm.invoke("Who are you")
            self.assertIsNotNone(result)
            self.assertTrue(isinstance(result, AIMessage))
            logger.info(f"{id_or_path} result: {result.content}")
    #
    # @patch('langchain_google_genai.ChatGoogleGenerativeAI._generate')
    # def test_when_cm_invoke_failed(self, mock_generate):
    #     mcm = self.cm_pool.get_chat_model(self.cm_key)
    #     self.assertIsNotNone(mcm)
    #     self.assertTrue(isinstance(mcm, MultiChatModel))
    #     self.assertTrue(all([mw.ready for mw in mcm.cms]))
    #     mock_generate.side_effect = Exception("Mocked exception")
    #     self.assertRaises(Exception, mcm.invoke, "Who are you")
    #     self.assertTrue(all([mw.ready for mw in mcm.cms]))

    # def test_multi_invoke(self):
    #     questions = ['Who are you',
    #                  'What is your name',
    #                  'What is your purpose',
    #                  'What is your favorite color',
    #                  'What is your favorite food']
    #
    #     mp.set_start_method('spawn', force=True)
    #     n_workers = len(questions)
    #     # with mp.Pool(processes=n_workers) as pool:
    #     #     results = pool.starmap(self.subproc_invoke, zip(np.array_split(questions, n_workers)))
    #     pool = ThreadPool(n_workers)
    #     results = pool.map(self.subproc_invoke, questions)
    #     self.assertEqual(len(results), n_workers)
    #
    # def subproc_invoke(self, question):
    #     cm = self.cm_registry.get_chat_model(self.cm_key)
    #     result = cm.invoke(question)
    #     print(result.content)
    #     return result


if __name__ == '__main__':
    unittest.main()
