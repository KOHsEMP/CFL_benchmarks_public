#!/bin/bash

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


base_name='adult_CAll_base'

method="free"

seed_list=(42 43 44 45 46)
use_bar_feature_list=(True False)

default_obj_lam_list=(1.0 0.1 0.01 0.001) #(5.0 10.0 100.0)
if ["$#" -gt 0]; then
    obj_lam_list=("$@")
else
    obj_lam_list=("${default_obj_lam_list[@]}")
fi

pred_loss_func='log'

config_dir='config'
base_yaml="${config_dir}/${base_name}.yaml"

for use_bar_feature in ${use_bar_feature_list[@]}; do
    for obj_lam in ${obj_lam_list[@]}; do
        for seed in ${seed_list[@]}; do
            tmp_yaml="${config_dir}/tmp_joint_${base_name}_${method}_${seed}_${use_bar_feature}.yaml"
            cp ${base_yaml} ${tmp_yaml}
            echo "" >> ${tmp_yaml}
            echo "method: ${method}" >> ${tmp_yaml}
            echo "seed: ${seed}" >> ${tmp_yaml}
            echo "use_bar_feature: ${use_bar_feature}" >> ${tmp_yaml}
            
            echo "is_joint: True" >> ${tmp_yaml}
            echo "obj_lam: ${obj_lam}" >> ${tmp_yaml}
            echo "pred_loss_func: ${pred_loss_func}" >> ${tmp_yaml}

            python joint_learning.py --config_file ${tmp_yaml}
            rm ${tmp_yaml}
        done
    done
done