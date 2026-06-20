#!/bin/bash
PROJECT_DIR="/projects/prjs2127/airevolve/results"
SCRATCH_BASE="/scratch-shared/$USER"
RESUME_JOB="23983075"

declare -A CRASHED_RUNS=(
    ["2"]="exp_standard_ppo_power_ea_figure8_rep2"
    ["4"]="exp_standard_ppo_power_ea_figure8_rep4"
    ["5"]="exp_standard_ppo_power_ea_figure8_rep5"
    ["8"]="exp_standard_ppo_power_ea_shuttlerun_rep3"
    ["9"]="exp_standard_ppo_power_ea_shuttlerun_rep4"
)

for TASK_ID in "${!CRASHED_RUNS[@]}"; do
    RUN_NAME="${CRASHED_RUNS[$TASK_ID]}"
    RESUME_SCRATCH="${SCRATCH_BASE}/airevolve_resume_${RESUME_JOB}_${TASK_ID}/results/${RUN_NAME}_resumed"
    TARGET_RL_LOGS="${PROJECT_DIR}/${RUN_NAME}/rl_logs/generation_40"

    # Make sure the target directory exists and is empty
    mkdir -p "$TARGET_RL_LOGS"
    rm -rf "${TARGET_RL_LOGS}/"*

    if [ -d "${RESUME_SCRATCH}/rl_logs/generation_01" ]; then
        echo "Fixing IDs for ${RUN_NAME}..."
        
        # Loop through 0024 to 0055 and rename them to 1272 to 1303
        GLOBAL_ID=1272
        for LOCAL_ID in {24..55}; local_formatted=$(printf "%04d" $LOCAL_ID); do
            SOURCE_FOLDER="${RESUME_SCRATCH}/rl_logs/generation_01/individual_${local_formatted}"
            if [ -d "$SOURCE_FOLDER" ]; then
                global_formatted=$(printf "%04d" $GLOBAL_ID)
                cp -r "$SOURCE_FOLDER" "${TARGET_RL_LOGS}/individual_${global_formatted}"
            fi
            ((GLOBAL_ID++))
        done
        echo "  -> Renamed and copied 32 true Gen 40 individuals."
    else
        echo "  [Error] Could not find scratch data for ${RUN_NAME}"
    fi
done