#!/bin/bash

mkdir -p ./logs

nohup python -u preprocess/decomposition.py > ./logs/decomposition.log 2>&1 &