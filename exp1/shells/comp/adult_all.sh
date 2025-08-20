#!/bin/bash

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


base_name='adult_CAll_base'

seed_list=(42 43 44 45 46)


method="comp"

config_dir='config'
base_yaml="${config_dir}/${base_name}.yaml"

for seed in ${seed_list[@]}
do
    tmp_yaml="${config_dir}/tmp_${base_name}_${method}_${seed}.yaml"
    cp ${base_yaml} ${tmp_yaml}
    echo "" >> ${tmp_yaml}
    echo "method: ${method}" >> ${tmp_yaml}
    echo "seed: ${seed}" >> ${tmp_yaml}

    python main.py --config_file ${tmp_yaml}
    rm ${tmp_yaml}

done