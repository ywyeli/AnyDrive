#!/bin/bash

set -e

cd ../scenario_runner

while IFS= read -r line; do
  python scenario_runner.py \
  --route srunner/data/nus_berkeley.xml srunner/data/no_scenarios.json "$line" \
  --agent srunner/autoagents/npc_agent.py \
  --lidar-params ../hyperparams/det_6x60.toml \
  --hyperparams ../hyperparams/det_6x60.toml \
  --split training \
  --full_round &>> ../logs/det_6x60.out
done < ../scripts/berkeley_routes.txt

cd ../scripts

