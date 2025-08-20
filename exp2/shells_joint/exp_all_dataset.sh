#!/bin/bash

method=$1

dataset_list=(
            "adult"
            "bank"
)

for dataset in "${dataset_list[@]}"; do
    bash "./shells_joint/${method}/${dataset}_all.sh"
done