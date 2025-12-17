import logging.config

import uvicorn
import argparse
from fastapi import FastAPI, HTTPException, Request
from typing import List

from pmmgr.cm import ChatModelPool
from pmmgr.common import settings, MessageCode, FileUtils
import pmmgr.db as db
from pmmgr.db import PretrainedModelInfo

# Logger
logging.config.fileConfig(settings.LOGGING_CONFIG_PATH)
logger = logging.getLogger('pmmgr')
app = FastAPI()

DB_ERROR = MessageCode.DB_ERROR
NO_AVAILABLE_CHAT_MODEL = MessageCode.NO_AVAILABLE_CHAT_MODEL
UNKNOWN_ERROR = MessageCode.UNKNOWN_ERROR

@app.get("/")
def root_as():
    return "ChatModel Web Service"


@app.get("/model/chat/{id_or_path}/{query}", response_model=str)
def generate_answer(id_or_path: str, query: str):
    try:
        logger.info(f"Call generate_answer with id_or_path: {id_or_path}, {query}")
        cm = ChatModelPool.get_instance().get_chat_model(id_or_path)
        result = cm.invoke(query)
        return result.content if result else ""

    except Exception as e:
        logger.error(f"Error occurred while processing chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/info/", response_model=PretrainedModelInfo)
def add_pretrained_model_info(model_info: PretrainedModelInfo):
    try:
        db.insert_pretrained_model_info(model_info=model_info)
        return model_info
    except Exception as e:
        logger.error(f"Error occurred while adding model info: {e}")
        raise HTTPException(status_code=400, detail=MessageCode.get_message(DB_ERROR))

@app.get("/model/info/{id_or_path}", response_model=PretrainedModelInfo)
def get_pretrained_model_info(id_or_path: str):
    try:
        model_info = db.get_pretrained_model_info(id_or_path=id_or_path)
        return model_info
    except Exception as e:
        logger.error(f"Error occurred while getting model info: {e}")
        raise HTTPException(status_code=400, detail=MessageCode.get_message(DB_ERROR))

@app.get("/model/info/", response_model=List[PretrainedModelInfo])
def list_pretrained_model_infos():
    try:
        return db.list_pretrained_model_info()
    except Exception as e:
        logger.error(f"Error occurred while listing model infos: {e}")
        raise HTTPException(status_code=400, detail=MessageCode.get_message(DB_ERROR))

@app.put("/model/info/", response_model=PretrainedModelInfo)
def update_pretrained_model_info(model_info: PretrainedModelInfo):
    try:
        db.update_pretrained_model_info(model_info=model_info)
        return model_info
    except Exception as e:
        logger.error(f"Error occurred while updating model info: {e}")
        raise HTTPException(status_code=400, detail=MessageCode.get_message(DB_ERROR))

@app.delete("/model/info/{id_or_path}", response_model=dict)
def delete_pretrained_model(id_or_path: str):
    try:
        db.delete_pretrained_model_info(id_or_path=id_or_path)
        return {"detail": f"Model {id_or_path} deleted successfully"}
    except Exception as e:
        logger.error(f"Error occurred while deleting model info: {e}")
        raise HTTPException(status_code=400, detail=MessageCode.get_message(DB_ERROR))


def run_server(args):
    logger.info(f"Starting Web Server with args: {args}")
    uvicorn.run(app, host=args.host, port=args.port)

def init_pminfo(args):
    logger.info(f"Initializing pretrained model info DB with args: {args}")
    db.init_pretrained_model_info(csv_file=args.csv_file)
    logger.info(f"Pretrained model info DB initialized successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrained Model Manager")
    parser.add_argument('--log_level', type=str, default=settings.LOG_LEVEL)
    subparsers = parser.add_subparsers()

    # Arguments for sub command 'run_server'
    sub_parser = subparsers.add_parser('run_server', help='Run the server')
    sub_parser.set_defaults(func=run_server)
    sub_parser.add_argument("--host", type=str, default='0.0.0.0')
    sub_parser.add_argument("--port", type=int, default=8001)

    # Arguments for sub command 'init_pminfo'
    sub_parser = subparsers.add_parser('init_pminfo', help='Initialize pretrained model info DB')
    sub_parser.set_defaults(func=init_pminfo)
    sub_parser.add_argument("--csv_file", type=str, default=settings.PM_INFO_CSV)

    args = parser.parse_args()
    print('Logging level: %s' % args.log_level)
    logger.setLevel(args.log_level)
    args.func(args)
