import uvicorn
import argparse
from fastapi import FastAPI, HTTPException
from typing import List
from pmmgr.common import settings, logger
import pmmgr.db as db
from pmmgr.db import PretrainedModel

app = FastAPI()

@app.post("/models/", response_model=PretrainedModel)
def add_pretrained_model(model: PretrainedModel):
    try:
        db.insert_pretrained_model(model=model)
        return model
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/models/{mid}", response_model=PretrainedModel)
def get_pretrained_model(mid: str):
    model = db.get_pretrained_model(mid=mid)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return model

@app.get("/models/", response_model=List[PretrainedModel])
def get_all_pretrained_models():
    return db.get_all_pretrained_models()

@app.put("/models/{mid}", response_model=PretrainedModel)
def update_pretrained_model(mid: str, model: PretrainedModel):
    existing_model = db.get_pretrained_model(mid=mid)
    if existing_model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    
    db.update_pretrained_model(model=model)
    return model

@app.delete("/models/{mid}", response_model=dict)
def delete_pretrained_model(mid: str):
    existing_model = db.get_pretrained_model(mid=mid)
    if existing_model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    
    db.delete_pretrained_model(mid=mid)
    return {"detail": "Model deleted successfully"}

def main(args):
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start PMMgr Web Service")
    parser.add_argument('--log_level', type=str, default=settings.LOG_LEVEL)
    parser.add_argument("--port", type=int, default=8001)
    
    args = parser.parse_args()
    logger.setLevel(args.log_level)
    main(args)
