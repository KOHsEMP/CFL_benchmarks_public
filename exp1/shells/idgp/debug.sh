#!/bin/bash

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


base_name='adult_debug'

seed_list=(42)

method="idgp"

idgp_lr_g=0.00005
idgp_wd_g=0.0002
idgp_warmup_ep=5
idgp_T_1=4.0
idgp_T_2=4.0
idgp_alpha_list=(100.0)
idgp_beta=0.02
idgp_delta=1.0
idgp_theta=0.1
idgp_gamma=1.0
idgp_eta=1.0


config_dir='config'
base_yaml="${config_dir}/${base_name}.yaml"

for idgp_alpha in ${idgp_alpha_list[@]}
do
    for seed in ${seed_list[@]}
    do
        tmp_yaml="${config_dir}/tmp_${base_name}_${method}_${idgp_lr_g}_${idgp_wd_g}_${idgp_warmup_ep}_${idgp_T_1}_${idgp_T_2}_${idgp_alpha}_${idgp_beta}_${idgp_delta}_${idgp_theta}_${idgp_gamma}_${idgp_eta}_${seed}.yaml"
        cp ${base_yaml} ${tmp_yaml}
        echo "" >> ${tmp_yaml}
        echo "method: ${method}" >> ${tmp_yaml}
        echo "seed: ${seed}" >> ${tmp_yaml}
        echo "idgp_lr_g: ${idgp_lr_g}" >> ${tmp_yaml}
        echo "idgp_wd_g: ${idgp_wd_g}" >> ${tmp_yaml}
        echo "idgp_warmup_ep: ${idgp_warmup_ep}" >> ${tmp_yaml}
        echo "idgp_T_1: ${idgp_T_1}" >> ${tmp_yaml}
        echo "idgp_T_2: ${idgp_T_2}" >> ${tmp_yaml}
        echo "idgp_alpha: ${idgp_alpha}" >> ${tmp_yaml}
        echo "idgp_beta: ${idgp_beta}" >> ${tmp_yaml}
        echo "idgp_delta: ${idgp_delta}" >> ${tmp_yaml}
        echo "idgp_theta: ${idgp_theta}" >> ${tmp_yaml}
        echo "idgp_gamma: ${idgp_gamma}" >> ${tmp_yaml}
        echo "idgp_eta: ${idgp_eta}" >> ${tmp_yaml}

        python main.py --config_file ${tmp_yaml}
        rm ${tmp_yaml}
    done
done
