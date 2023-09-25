#!/bin/bash

# Accept --load-dir and --num-timesteps as arguments 
# --load-dir is the path to the last checkpoint of your previous run
# --num-timesteps is the number of timesteps you want to train for

# Check if the number of arguments is correct
if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ./task_2.1_baseline_agent_random.sh <path_to_last_checkpoint> <num_timesteps>"
    exit
fi

LOAD_DIR=$1 # <File Path to your last Checkpoint> \
NUM_TIMESTEPS=$2 # <Update this if you already reached 1M timesteps from your previous run. For example, extends it to 2e6 >


python multigrid/scripts/train.py \
 --local-mode False \
 --env MultiGrid-CompetativeRedBlueDoor-v3-CTDE-Red \
 --num-workers 10 \
 --num-gpus 0 \
 --name CTDE-Red_baseline \
 --training-scheme CTDE  \
 --training-config '{"team_policies_mapping": {"red_0" : "your_policy_name" , "red_1" : "your_policy_name_v2" }}' \
 --restore-all-policies-from-checkpoint True \
 --policies-to-load red_0 red_1 \
 --load-dir $LOAD_DIR \
 --num-timesteps $NUM_TIMESTEPS 