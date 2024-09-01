#!/bin/bash

FLASK_SERVER_SCRIPT="app.py"
DATA_PREP="src/data_prep.py"
MODEL_TRAINING="src/model_training.py"
ANOMALY_DETECTION="src/anomaly_detection.py"

# start flask server in the background
nohup python3 $FLASK_SERVER_SCRIPT &

# wait for flask server to start
sleep 5

# run data preparation script in the backgrround
nohup python3 $DATA_PREP &

# run model training script in the background
nohup python3 $MODEL_TRAINING &

# run anomaly detection script
python3 $ANOMALY_DETECTION