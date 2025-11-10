#!/bin/bash

MODEL_IDS="deepseek-r1:1.5b gemma3:1b llama3.2:1b qwen2.5:0.5b phi3:3.8b qwen2.5-coder:0.5b tinyllama:1.1b starcoder2:3b granite3.1-moe:1b falcon3:1b"
QUERY="Tell me about RAG(Retrival Augmented Generation) with a paragraph of three sentences"
OUTFILE="./answer.txt"

for mid in $MODEL_IDS; do
    echo "=== Generating response using $mid ==="
    echo "$mid:" >> $OUTFILE
    ollama run $mid "$QUERY" >> $OUTFILE
#    echo -e "\n" >> $OUTFILE
done