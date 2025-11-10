import csv

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import json

from pmmgr.common import settings, TaskType, QuantType, IOType, LicenseType, ModelType, EvalMetric, BaseTest
import unittest
import sqlite3
import os
import logging

###
# Logger
logger = logging.getLogger('pmmgr')

class PretrainedModelInfo(BaseModel):
    id_or_path: str  # primary key
    name: str = Field(default="NO_NAME")
    created_at: datetime = Field(default_factory=datetime.now)
    task_types: str = Field(default=TaskType.TEXT_GENERATION) 
    description: Optional[str] = None
    n_params: int = Field(default=0)
    input_type: str = Field(default=IOType.TEXT)
    output_type: str = Field(default=IOType.TEXT)
    quant_type: str = Field(default=QuantType.FP16)
    size: int = Field(default=0)
    last_modified_at: datetime = Field(default_factory=datetime.now)
    mtype: str = Field(default=ModelType.OLLAMA)
    context_len: int = Field(default=0)
    license: Optional[str] = Field(default=LicenseType.APACHE20)
    n_srv_instances: int = Field(default=1)
    config: dict = Field(default_factory=dict)
    eval_metric: Optional[str] = Field(default=EvalMetric.AVG_MR_GPQA)
    eval_score: Optional[float] = Field(default=0.0)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return (f"PretrainedModelInfo(id_or_path='{self.id_or_path}', name='{self.name}', description='{self.description}', "
                f"mtype='{self.mtype}', context_len={self.context_len}, license='{self.license}', n_srv_instances={self.n_srv_instances}, "
                f"config={self.config}, eval_metric='{self.eval_metric}', eval_score={self.eval_score})")

    def __eq__(self, other):
        return self.id_or_path == other.id_or_path

    @classmethod
    def from_row_dict(cls, row_dict: dict):
        created_at = row_dict.get('created_at', datetime.now())
        if created_at and isinstance(created_at, str):
            created_at = datetime.strptime(created_at, '%Y-%m-%d %H:%M:%S')
        last_modified_at = row_dict.get('last_modified_at', datetime.now())
        if last_modified_at and isinstance(last_modified_at, str):
            last_modified_at = datetime.strptime(last_modified_at, '%Y-%m-%d %H:%M:%S')
        config = row_dict.get('config', {})
        if config and isinstance(config, str):
            config = json.loads(config)
        n_params = row_dict.get('n_params', 0)
        if n_params and isinstance(n_params, str):
            n_params = int(n_params)
        msize = row_dict.get('size', 0)
        if msize and isinstance(msize, str):
            msize = int(msize)
        context_len = row_dict.get('context_len', 0)
        if context_len and isinstance(context_len, str):
            context_len = int(context_len)
        n_srv_instances = row_dict.get('n_srv_instances', 0)
        if n_srv_instances and isinstance(n_srv_instances, str):
            n_srv_instances = int(n_srv_instances)
        eval_score = row_dict.get('eval_score', 0.0)
        if eval_score and isinstance(eval_score, str):
            eval_score = float(eval_score)

        return cls(
            id_or_path=row_dict['id_or_path'],
            name=row_dict.get('name', 'NO_NAME'),
            task_types=row_dict.get('task_types', TaskType.TEXT_GENERATION),
            description=row_dict.get('description'),
            n_params=n_params,
            input_type=row_dict.get('input_type', IOType.TEXT),
            output_type=row_dict.get('output_type', IOType.TEXT),
            quant_type=row_dict.get('quant_type', QuantType.NONE),
            size=msize,
            created_at=created_at,
            last_modified_at=last_modified_at,
            mtype=row_dict.get('mtype', ModelType.OLLAMA),
            context_len=context_len,
            license=row_dict.get('license', LicenseType.APACHE20),
            n_srv_instances = n_srv_instances,
            config = config,
            eval_metric=row_dict.get('eval_metric', EvalMetric.AVG_MR_GPQA),
            eval_score=eval_score
        )

    def to_dict(self) -> dict:
        return {
            'id_or_path': self.id_or_path,
            'name': self.name,
            'created_at': self.created_at,
            'task_types': self.task_types,
            'description': self.description,
            'n_params': self.n_params,
            'input_type': self.input_type,
            'output_type': self.output_type,
            'quant_type': self.quant_type,
            'size': self.size,
            'last_modified_at': self.last_modified_at,
            'mtype': self.mtype,
            'context_len': self.context_len,
            'license': self.license,
            'n_srv_instances': self.n_srv_instances,
            'config': self.config,
            'eval_metric': self.eval_metric,
            'eval_score': self.eval_score
        }


def create_pretrained_model_info_table(db_path: str = settings.DB_PATH):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE "PretrainedModelInfo"
            (
                id_or_path  TEXT  not null primary key,
                name       TEXT  default "NO_NAME",
                created_at  TEXT  default (datetime('now', 'localtime')) not null,
                task_types  TEXT  default "Text Generation" not null,
                description TEXT,
                n_params    INTEGER default 0 not null,
                input_type  TEXT default "Text" not null,
                output_type TEXT default "Text" not null,
                quant_type  TEXT default "-" not null,
                size        INTEGER default 0,
                last_modified_at TEXT default (datetime('now', 'localtime')) not null,
                mtype       TEXT default "OLLAMA",
                context_len INTEGER default 0,
                license     TEXT default "Apache-2.0",
                n_srv_instances INTEGER default 1,
                config TEXT default "{}",
                eval_metric TEXT default "Average of MMLU-Redux, GPQA-Diamond",
                eval_score REAL default 0.0
            )
        ''')
        conn.commit()


def insert_pretrained_model_info(db_path: str = settings.DB_PATH, model_info: PretrainedModelInfo = None):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        model_dict = model_info.to_dict()
        model_dict['created_at'] = model_info.created_at.strftime('%Y-%m-%d %H:%M:%S')
        model_dict['last_modified_at'] = model_info.last_modified_at.strftime('%Y-%m-%d %H:%M:%S')
        model_dict['config'] = json.dumps(model_info.config or {})

        cursor.execute('''
            INSERT INTO PretrainedModelInfo (id_or_path, name, created_at, task_types, description, 
                   n_params, input_type, output_type, quant_type, size, last_modified_at,
                   mtype, context_len, license, n_srv_instances, config, eval_metric, eval_score)
            VALUES (:id_or_path, :name, :created_at, :task_types, :description, 
                   :n_params, :input_type, :output_type, :quant_type, :size,
                   :last_modified_at, :mtype, :context_len, :license, :n_srv_instances, :config, :eval_metric, :eval_score)
            ''',
                       model_dict
                       )
        conn.commit()

def get_pretrained_model_info(db_path: str = settings.DB_PATH, id_or_path: str = None) -> Optional['PretrainedModelInfo']:
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM PretrainedModelInfo WHERE id_or_path = ?', (id_or_path,))
        row = cursor.fetchone()
        if row:
            column_names = [description[0] for description in cursor.description]
            row_dict = dict(zip(column_names, row))
            return PretrainedModelInfo.from_row_dict(row_dict)
        return None

def list_pretrained_model_info(db_path: str = settings.DB_PATH) -> List['PretrainedModelInfo']:
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM PretrainedModelInfo')
        rows = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]
        return [PretrainedModelInfo.from_row_dict(dict(zip(column_names, row))) for row in rows]

def update_pretrained_model_info(db_path: str = settings.DB_PATH, model_info: PretrainedModelInfo = None):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        model_dict = model_info.to_dict()
        model_dict['last_modified_at'] = model_info.last_modified_at.strftime('%Y-%m-%d %H:%M:%S')
        model_dict['config'] = json.dumps(model_info.config or {})

        cursor.execute('''
                UPDATE PretrainedModelInfo 
                SET name=:name, task_types=:task_types, description=:description,
                    n_params=:n_params, input_type=:input_type, output_type=:output_type,
                    quant_type=:quant_type, size=:size, last_modified_at=:last_modified_at,
                    mtype=:mtype, context_len=:context_len, license=:license, n_srv_instances=:n_srv_instances, config=:config,
                    eval_metric=:eval_metric, eval_score=:eval_score
                WHERE id_or_path=:id_or_path
            ''', model_dict)
        conn.commit()

def delete_pretrained_model_info(db_path: str = settings.DB_PATH, id_or_path: str = None):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM PretrainedModelInfo WHERE id_or_path = ?', (id_or_path,))
        conn.commit()


def delete_all_pretrained_model_info(db_path: str = settings.DB_PATH):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM PretrainedModelInfo')
        conn.commit()


def init_pretrained_model_info(db_path: str = settings.DB_PATH, csv_file: str = None):
    with sqlite3.connect(db_path) as conn:
        with open(csv_file, 'r', encoding='utf-8') as f:
            delete_all_pretrained_model_info(db_path=db_path)

            csv_reader = csv.DictReader(f)
            for row in csv_reader:
                logger.info(f"Inserting pretrained model info: {row}")
                insert_pretrained_model_info(db_path=db_path,
                                             model_info=PretrainedModelInfo.from_row_dict(row))



class DBTest(BaseTest):
    def setUp(self):
        super().setUp()

        if os.path.exists(settings.TEST_DB_PATH):
            os.remove(settings.TEST_DB_PATH)
        create_pretrained_model_info_table(settings.TEST_DB_PATH)
        
    def tearDown(self):
        if os.path.exists(settings.TEST_DB_PATH):
            os.remove(settings.TEST_DB_PATH)

    def test_insert_and_get_pretrained_model(self):
        test_model = PretrainedModelInfo(
            id_or_path="test123",
            name="Test Model",
            description="Test Description",
            n_params=1000000,
            size=500,
            context_len=4096
        )
        insert_pretrained_model_info(settings.TEST_DB_PATH, test_model)
        retrieved_model = get_pretrained_model_info(settings.TEST_DB_PATH, "test123")
        self.assertEqual(retrieved_model.id_or_path, test_model.id_or_path)
        self.assertEqual(retrieved_model.name, test_model.name)
        self.assertEqual(retrieved_model.n_params, test_model.n_params)

    def test_get_all_pretrained_models(self):
        test_model = PretrainedModelInfo(
            id_or_path="test123",
            name="Test Model",
            description="Test Description",
            n_params=1000000,
            size=500,
            context_len=4096
        )
        insert_pretrained_model_info(settings.TEST_DB_PATH, test_model)
        all_models = list_pretrained_model_info(settings.TEST_DB_PATH)
        self.assertEqual(len(all_models), 1)
        self.assertEqual(all_models[0].id_or_path, test_model.id_or_path)
        self.assertEqual(all_models[0].name, test_model.name)
        self.assertEqual(all_models[0].n_params, test_model.n_params)

    def test_update_pretrained_model(self):
        test_model = PretrainedModelInfo(
            id_or_path="test123",
            name="Test Model",
            description="Test Description",
            n_params=1000000,
            size=500,
            context_len=4096
        )
        insert_pretrained_model_info(settings.TEST_DB_PATH, test_model)

        updated_model = PretrainedModelInfo(
            id_or_path="test123",
            name="Updated Model",
            description="Updated Description",
        )
        update_pretrained_model_info(settings.TEST_DB_PATH, updated_model)
        retrieved_model = get_pretrained_model_info(settings.TEST_DB_PATH, "test123")
        self.assertEqual(retrieved_model.name, updated_model.name)
        self.assertEqual(retrieved_model.description, updated_model.description)    
    
    def test_delete_pretrained_model(self):
        test_model = PretrainedModelInfo(
            id_or_path="test123",
            name="Test Model",
        )   
        insert_pretrained_model_info(settings.TEST_DB_PATH, test_model)
        delete_pretrained_model_info(settings.TEST_DB_PATH, "test123")
        retrieved_model = get_pretrained_model_info(settings.TEST_DB_PATH, "test123")
        self.assertIsNone(retrieved_model)  


if __name__ == '__main__':
    unittest.main()

