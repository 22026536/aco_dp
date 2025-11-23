#!/bin/bash

rm -rf results/logs/dp_logs
mkdir -p results/logs/dp_logs/evolution
mkdir -p results/logs/dp_logs/solutions
mkdir -p results/logs/dp_logs/objectives

config_params='--schema 2P-GRASP-DP --version rnd-grd --m 8 --block 38 --delta 5 --exploration first --debug 0'
fixed_params='--termination_criteria tcpu --termination_value 300 --logs 1 --move ext --efficient 1'

instance_path="$1"
if [ -z "$instance_path" ]; then
  echo "❌ Instance path for GRASP-DP not found!"
  exit 1
fi

./MCGP --instance "$instance_path" --seed 998244353 $fixed_params $config_params
