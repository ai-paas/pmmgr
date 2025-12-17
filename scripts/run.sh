#!/bin/bash

ROOT_DIR=$(cd $(dirname $0)/..; pwd)
PORT=8001

export PATH=$PATH:$ROOT_DIR:$ROOT_DIR/pmmgr
export PYTHONPATH=$PYTHONPATH:$ROOT_DIR:$ROOT_DIR/pmmgr

# conda activate PMMgr

python $ROOT_DIR/pmmgr/main.py run_server --port $PORT
