#!/bin/bash

# If there is no argument, then exit with error
if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ./task_3_visualize_agent.sh <path_to_last_checkpoint>"
    echo "(submission/ray_results/PPO/PPO_MultiGrid-CompetativeRedBlueDoor-v3-DTDE-1v1_XXXX/checkpoint_YYY/checkpoint-YYY)"
    exit
fi

LOAD_DIR=$1

python multigrid/scripts/visualize.py \
 --env MultiGrid-CompetativeRedBlueDoor-v3-DTDE-1v1 \
 --num-episodes 10 \
 --load-dir $LOAD_DIR \
 --render-mode rgb_array \
 --gif DTDE_1v1-testing \
 --team-policies-mapping '{"red_0" : "pickupper"}}' \
 --policies-to-eval red_0 