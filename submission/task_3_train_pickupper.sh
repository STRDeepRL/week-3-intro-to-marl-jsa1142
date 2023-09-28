#!/bin/bash

nohup python multigrid/scripts/train.py \
  --env MultiGrid-CompetativeRedBlueDoor-v3-DTDE-1v1 \
  --local-mode False \
  --num-workers 23 \
  --num-gpus 0 \
  --name pickupper \
  --training-scheme DTDE \
  --training-config '{"team_policies_mapping": {"red_0" : "pickupper"}}' \
  --restore-all-policies-from-checkpoint False \
  --using-self-play True \
  --seed 1 \
  --win-rate-threshold 0.85 \
  > "./submission/logs/self_play_pickupper.log" 2>&1 &
