#!/bin/bash

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


base_name='bank_CAll_base'

seed_list=(42 43 44 45 46)

method="free"


config_dir='config'
base_yaml="${config_dir}/${base_name}.yaml"


for seed in ${seed_list[@]}
do
    tmp_name="tmp_${base_name}_${method}_${seed}"
    tmp_dir="${config_dir}/${tmp_name}"
    if [ ! -e ${tmp_dir} ]; then mkdir ${tmp_dir} ; fi

    tmp_yaml="${config_dir}/${tmp_name}/${tmp_name}.yaml"
    cp ${base_yaml} ${tmp_yaml}
    echo "" >> ${tmp_yaml}
    echo "method: ${method}" >> ${tmp_yaml}
    echo "seed: ${seed}" >> ${tmp_yaml}

    # execute 'main.py' ==========================================================================
    # 1. use_bar_feature=False
    tmp_ubfF_yaml="${config_dir}/${tmp_name}/${tmp_name}_ubfF.yaml"
    cp ${tmp_yaml} ${tmp_ubfF_yaml}
    echo "use_bar_feature: False" >> ${tmp_ubfF_yaml}
    python main.py --config_file ${tmp_ubfF_yaml}
    python pred_avoid_est.py --config_file ${tmp_ubfF_yaml}


    # 2. use_bar_feature=True
    tmp_ubfT_yaml="${config_dir}/${tmp_name}/${tmp_name}_ubfT.yaml"
    cp ${tmp_yaml} ${tmp_ubfT_yaml}
    echo "use_bar_feature: True" >> ${tmp_ubfT_yaml}
    python main.py --config_file ${tmp_ubfT_yaml}
    python pred_avoid_est.py --config_file ${tmp_ubfT_yaml}


    # execute 'main_iter.py' =====================================================================
    # 3. use_bar_feature=False -> iter_avoid_est=[]
    tmp_ubfF_iter1_yaml="${config_dir}/${tmp_name}/${tmp_name}_ubfF_iter1.yaml"
    cp ${tmp_ubfF_yaml} ${tmp_ubfF_iter1_yaml}
    echo "iter_idx: 1" >> ${tmp_ubfF_iter1_yaml}
    python main_iter.py --config_file ${tmp_ubfF_iter1_yaml}
    python pred_avoid_est.py --config_file ${tmp_ubfF_iter1_yaml}


    # 4. use_bar_feature=False -> iter_avoid_est=[...]
    tmp_ubfF_iter1_AEC_yaml="${config_dir}/${tmp_name}/${tmp_name}_ubfF_iter1_AEC.yaml"
    cp ${tmp_ubfF_iter1_yaml} ${tmp_ubfF_iter1_AEC_yaml}
    echo "iter_aec: True" >> ${tmp_ubfF_iter1_AEC_yaml}
    python main_iter.py --config_file ${tmp_ubfF_iter1_AEC_yaml}
    python pred_avoid_est.py --config_file ${tmp_ubfF_iter1_AEC_yaml}


    # 5. use_bar_feature=True -> iter_avoid_est=[]
    tmp_ubfT_iter1_yaml="${config_dir}/${tmp_name}/${tmp_name}_ubfT_iter1.yaml"
    cp ${tmp_ubfT_yaml} ${tmp_ubfT_iter1_yaml}
    echo "iter_idx: 1" >> ${tmp_ubfT_iter1_yaml}
    python main_iter.py --config_file ${tmp_ubfT_iter1_yaml}
    python pred_avoid_est.py --config_file ${tmp_ubfF_iter1_AEC_yaml}


    # 6. use_bar_feature=True -> iter_avoid_est=[...]
    tmp_ubfT_iter1_AEC_yaml="${config_dir}/${tmp_name}/${tmp_name}_ubfT_iter1_AEC.yaml"
    cp ${tmp_ubfT_iter1_yaml} ${tmp_ubfT_iter1_AEC_yaml}
    echo "iter_aec: True" >> ${tmp_ubfT_iter1_AEC_yaml}
    python main_iter.py --config_file ${tmp_ubfT_iter1_AEC_yaml}
    python pred_avoid_est.py --config_file ${tmp_ubfT_iter1_AEC_yaml}


    rm -r ${tmp_dir}
done
