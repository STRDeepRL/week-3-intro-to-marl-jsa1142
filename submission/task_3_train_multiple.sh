#!/bin/bash

# List of policy names
policy_names=("eliminator" "pickupper" "default")
# policy_names=("eliminator" "pickupper")

# Loop over each policy name
for policy in "${policy_names[@]}"; do
  echo "Training policy: $policy"
  # Set a timeout
  timeout 300s nohup python multigrid/scripts/train.py \
    --env MultiGrid-CompetativeRedBlueDoor-v3-DTDE-1v1 \
    --local-mode False \
    --num-workers 23 \
    --num-gpus 0 \
    --name "$policy" \
    --training-scheme DTDE \
    --training-config "{\"team_policies_mapping\": {\"red_0\" : \"$policy\" }}" \
    --restore-all-policies-from-checkpoint False \
    --using-self-play True \
    --win-rate-threshold 0.85 \
    > "./submission/logs/self_play_${policy}.log" 2>&1 &
  
  # Wait for the script to exit
  wait $!
done
