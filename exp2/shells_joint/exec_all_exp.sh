#!/bin/bash

method_list=(
            #"rc"
            #"cc"
            #"proden" 
            #"forward" 
            #"free"
            #"nn" 
            "ga"
)

dataset_list=(
            "adult"
            "bank"
)

SESSIONS=()
for method in "${method_list[@]}"; do
    #SESSIONS+=("${method}-exp2-joint:bash ./shells_joint/exp_all_dataset.sh ${method}")
    for dataset in "${dataset_list[@]}"; do
        SESSIONS+=("${method}-${dataset}-joint:bash ./shells_joint/exp_base.sh ${method} ${dataset}")
    done
done


# 各セッションを作成し、指定したコマンドを実行
echo "Starting tmux sessions..."
for entry in "${SESSIONS[@]}"; do
    IFS=":" read -r session_name command <<< "$entry"
    
    # 新しいtmuxセッションを作成（すでに存在する場合はスキップ）
    if ! tmux has-session -t "$session_name" 2>/dev/null; then
        tmux new-session -d -s "$session_name"
        echo "Created session: $session_name"
    fi
    
    # 指定のコマンドを実行
    tmux send-keys -t "$session_name" "$command" C-m
    echo "Started command in $session_name: $command"
done

echo "All tmux sessions started."
