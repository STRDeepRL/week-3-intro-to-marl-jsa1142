python multigrid/scripts/train.py \
 --local-mode False \
 --env MultiGrid-CompetativeRedBlueDoor-v3-CTDE-Red-NonRandom \
 --num-workers 10 \
 --num-gpus 0 \
 --name CTDE-Red_baseline \
 --training-scheme CTDE \
 --training-config '{"team_policies_mapping": {"red_0" : "your_policy_name" , "red_1" : "your_policy_name_v2" }}' \
 --restore-all-policies-from-checkpoint False