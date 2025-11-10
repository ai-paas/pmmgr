#!/bin/bash

ROOT_DIR=$(cd $(dirname $0)/..; pwd)
SERVICE=saju
#PORT=8000

export PATH=$PATH:$ROOT_DIR:$ROOT_DIR/chatcat
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR:$ROOT_DIR/chatcat

# conda activate ChatCat

#python $ROOT_DIR/chatcat/run.py --service $SERVICE --port $PORT
python $ROOT_DIR/chatcat/run.py --service $SERVICE