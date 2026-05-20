#!/bin/bash

instance_path="$1"
if [ -z "$instance_path" ]; then
  echo "❌ Instance path for ACO not found!"
  exit 1
fi

# Lấy tên instance (bỏ đường dẫn và phần mở rộng)
instance_name=$(basename "$instance_path" | sed 's/\.[^.]*$//')

rm -rf results/logs
mkdir -p "results/logs/${instance_name}/evolution"
mkdir -p "results/logs/${instance_name}/solutions"
mkdir -p "results/logs/${instance_name}/objectives"

config_params='--schema 2P-ACO-DP --version rnd-grd --m 8 --block 38 --delta 5 --exploration first --debug 0'

time_limit="${2:-1200}"

fixed_params="--termination_criteria tcpu --termination_value $time_limit --logs 1 --move ext --efficient 1"

./MCGP --instance "$instance_path" --seed 998244353 $fixed_params $config_params
