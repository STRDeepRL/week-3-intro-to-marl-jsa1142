#!/bin/bash

# If there is no argument, then exit with error
if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ./visualize_agent.sh <path_to_last_checkpoint>"
    exit
fi

LOAD_DIR=$1

python multigrid/scripts/visualize.py \
 --env MultiGrid-CompetativeRedBlueDoor-v3-CTDE-Red-NonRandom \
 --num-episodes 10 \
 --load-dir $LOAD_DIR \
 --render-mode rgb_array \
 --gif CTDE-Red-testing \
 --policies-to-eval red_0 red_1 \
 --eval-config '{"team_policies_mapping": {"red_0" : "your_policy_name" , "red_1" : "your_policy_name_v2" }}'