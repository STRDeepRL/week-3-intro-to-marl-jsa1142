#!/bin/bash

#  --local-mode False \

nohup python multigrid/scripts/train.py \
 --env MultiGrid-CompetativeRedBlueDoor-v3-DTDE-1v1 \
 --num-workers 20 \
 --num-gpus 0 \
 --name Policy_Self_Play_baseline \
 --training-scheme DTDE  \
 --training-config '{"team_policies_mapping": {"red_0" : "your_policy_name" }}' \
 --restore-all-policies-from-checkpoint False \
 --using-self-play True \
 --win-rate-threshold 0.85 \
 > ./submission/logs/Policy_Self_Play_baseline.log 2>&1 &