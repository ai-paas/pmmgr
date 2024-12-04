from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from pmmgr.common import settings, TaskType, QuantType, IOType
import unittest
import sqlite3
import os

class PretrainedModel(BaseModel):
    mid: str  # primary key
    name: str = Field(default="NO_NAME")
    model_path: str = Field(default="")
    created_at: datetime = Field(default_factory=datetime.now)
    task_types: str = Field(default=TaskType.TEXT_GENERATION) 
    description: Optional[str] = None
    n_params: int = Field(default=0)
    input_type: str = Field(default=IOType.TEXT)
    output_type: str = Field(default=IOType.TEXT)
    quant_type: str = Field(default=QuantType.FP16)
    config: Optional[str] = None
    vocab_size: int = Field(default=0)
    infer_code: Optional[str] = None
    last_modified_at: datetime = Field(default_factory=datetime.now)

    @classmethod
    def from_dict(cls, row: dict):
        created_at = datetime.strptime(row['created_at'], '%Y-%m-%d %H:%M:%S')
        last_modified_at = datetime.strptime(row['last_modified_at'], '%Y-%m-%d %H:%M:%S')
        
        return cls(
            mid=row['mid'],
            name=row['name'],
            model_path=row['model_path'],
            created_at=created_at,
            task_types=row['task_types'],
            description=row['description'],
            n_params=row['n_params'],
            input_type=row['input_type'],
            output_type=row['output_type'],
            quant_type=row['quant_type'],
            config=row['config'],
            vocab_size=row['vocab_size'],
            infer_code=row['infer_code'],
            last_modified_at=last_modified_at
        )

    def to_dict(self) -> dict:
        return {
            'mid': self.mid,
            'name': self.name,
            'model_path': self.model_path,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'task_types': self.task_types,
            'description': self.description,
            'n_params': self.n_params,
            'input_type': self.input_type,
            'output_type': self.output_type,
            'quant_type': self.quant_type,
            'config': self.config,
            'vocab_size': self.vocab_size,
            'infer_code': self.infer_code,
            'last_modified_at': self.last_modified_at.strftime('%Y-%m-%d %H:%M:%S')
        }


def create_pretrained_model_table(db_path: str = settings.DB_PATH):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE "PretrainedModel"
            (
                mid         TEXT  not null primary key,
                name        TEXT  default "NO_NAME",
                model_path  TEXT  default "" not null,
                created_at  TEXT  default (datetime('now', 'localtime')) not null,
                task_types  TEXT  default "Text Generation" not null,
                description TEXT,
                n_params    INTEGER default 0 not null,
                input_type  TEXT default "Text" not null,
                output_type TEXT default "Text" not null,
                quant_type  TEXT default "fp16" not null,
                config      TEXT,
                vocab_size  INTEGER default 0,
                infer_code  TEXT,
                last_modified_at TEXT default (datetime('now', 'localtime')) not null
)        ''')
        conn.commit()


def insert_pretrained_model(db_path: str = settings.DB_PATH, model: PretrainedModel = None):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO PretrainedModel (mid, name, model_path, created_at, task_types, description, 
                   n_params, input_type, output_type, quant_type, config, vocab_size, infer_code, last_modified_at)
            VALUES (:mid, :name, :model_path, :created_at, :task_types, :description, 
                   :n_params, :input_type, :output_type, :quant_type, :config,
                   :vocab_size, :infer_code, :last_modified_at)
            ''', 
            model.to_dict()
        )
        conn.commit()

def get_pretrained_model(db_path: str = settings.DB_PATH, mid: str = None) -> Optional['PretrainedModel']:
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM PretrainedModel WHERE mid = ?', (mid,))
        row = cursor.fetchone()
        if row:
            column_names = [description[0] for description in cursor.description]
            row_dict = dict(zip(column_names, row))
            return PretrainedModel.from_dict(row_dict)
        return None

def get_all_pretrained_models(db_path: str = settings.DB_PATH) -> List['PretrainedModel']:
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM PretrainedModel')
        rows = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]
        return [PretrainedModel.from_dict(dict(zip(column_names, row))) for row in rows]

def update_pretrained_model(db_path: str = settings.DB_PATH, model: PretrainedModel = None):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
                UPDATE PretrainedModel 
                SET name=:name, model_path=:model_path, task_types=:task_types, description=:description,
                    n_params=:n_params, input_type=:input_type, output_type=:output_type,
                    quant_type=:quant_type, config=:config, vocab_size=:vocab_size,
                    infer_code=:infer_code, last_modified_at=:last_modified_at
                WHERE mid=:mid
            ''', model.to_dict())
        conn.commit()

def delete_pretrained_model(db_path: str = settings.DB_PATH, mid: str = None):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('DELETE FROM PretrainedModel WHERE mid = ?', (mid,))
        conn.commit()

class DBTest(unittest.TestCase):
    def setUp(self):
        if os.path.exists(settings.TEST_DB_PATH):
            os.remove(settings.TEST_DB_PATH)
        
        create_pretrained_model_table(settings.TEST_DB_PATH)
        
    def tearDown(self):
        if os.path.exists(settings.TEST_DB_PATH):
            os.remove(settings.TEST_DB_PATH)

    def test_insert_and_get_pretrained_model(self):
        test_model = PretrainedModel(
            mid="test123",
            name="Test Model",
            model_path="test/model",
            description="Test Description",
            n_params=1000000,
            vocab_size=30000
        )
        insert_pretrained_model(settings.TEST_DB_PATH, test_model)
        retrieved_model = get_pretrained_model(settings.TEST_DB_PATH, "test123")
        self.assertEqual(retrieved_model.mid, test_model.mid)
        self.assertEqual(retrieved_model.name, test_model.name)
        self.assertEqual(retrieved_model.n_params, test_model.n_params)

    def test_get_all_pretrained_models(self):
        test_model = PretrainedModel(
            mid="test123",
            name="Test Model",
            model_path="test/model",
            description="Test Description",
            n_params=1000000,
            vocab_size=30000
        )
        insert_pretrained_model(settings.TEST_DB_PATH, test_model)
        all_models = get_all_pretrained_models(settings.TEST_DB_PATH)
        self.assertEqual(len(all_models), 1)
        self.assertEqual(all_models[0].mid, test_model.mid)
        self.assertEqual(all_models[0].name, test_model.name)
        self.assertEqual(all_models[0].n_params, test_model.n_params)

    def test_update_pretrained_model(self):
        test_model = PretrainedModel(
            mid="test123",
            name="Test Model",
            model_path="test/model",
            description="Test Description",
            n_params=1000000,
            vocab_size=30000
        )
        insert_pretrained_model(settings.TEST_DB_PATH, test_model)

        updated_model = PretrainedModel(
            mid="test123",
            name="Updated Model",
            model_path="test/model",
            description="Updated Description",
        )
        update_pretrained_model(settings.TEST_DB_PATH, updated_model)    
        retrieved_model = get_pretrained_model(settings.TEST_DB_PATH, "test123")
        self.assertEqual(retrieved_model.name, updated_model.name)
        self.assertEqual(retrieved_model.description, updated_model.description)    
    
    def test_delete_pretrained_model(self):
        test_model = PretrainedModel(
            mid="test123",
            name="Test Model",
            model_path="test/model",
        )   
        insert_pretrained_model(settings.TEST_DB_PATH, test_model)
        delete_pretrained_model(settings.TEST_DB_PATH, "test123")
        retrieved_model = get_pretrained_model(settings.TEST_DB_PATH, "test123")
        self.assertIsNone(retrieved_model)  


if __name__ == '__main__':
    unittest.main()