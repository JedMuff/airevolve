#!/bin/bash
PROJECT_DIR="/projects/prjs2127/airevolve/results"
SCRATCH_BASE="/scratch-shared/$USER"
ORIG_JOB="23732190"
RESUME_JOB="23983075"

# Map all tasks to their directory names
declare -A ALL_RUNS=(
    ["1"]="exp_standard_ppo_power_ea_figure8_rep1"
    ["2"]="exp_standard_ppo_power_ea_figure8_rep2"
    ["3"]="exp_standard_ppo_power_ea_figure8_rep3"
    ["4"]="exp_standard_ppo_power_ea_figure8_rep4"
    ["5"]="exp_standard_ppo_power_ea_figure8_rep5"
    ["6"]="exp_standard_ppo_power_ea_shuttlerun_rep1"
    ["7"]="exp_standard_ppo_power_ea_shuttlerun_rep2"
    ["8"]="exp_standard_ppo_power_ea_shuttlerun_rep3"
    ["9"]="exp_standard_ppo_power_ea_shuttlerun_rep4"
    ["10"]="exp_standard_ppo_power_ea_shuttlerun_rep5"
)

# Tasks that need the 40th generation grafted on
CRASHED_TASKS="2 4 5 8 9"

echo "Starting data unification..."

for TASK_ID in "${!ALL_RUNS[@]}"; do
    RUN_NAME="${ALL_RUNS[$TASK_ID]}"
    ORIG_SCRATCH="${SCRATCH_BASE}/airevolve_tmp_${ORIG_JOB}_${TASK_ID}/results/${RUN_NAME}"
    TARGET_PATH="${PROJECT_DIR}/${RUN_NAME}"

    echo "Processing Task ${TASK_ID} (${RUN_NAME})..."

    # 1. Sync the original data from scratch to the project dir
    if [ -d "$ORIG_SCRATCH" ]; then
        mkdir -p "$TARGET_PATH"
        rsync -a "${ORIG_SCRATCH}/" "${TARGET_PATH}/"
    else
        echo "  [Warning] Original scratch dir not found: $ORIG_SCRATCH"
    fi

    # 2. If it's a crashed run, graft the resumed data onto it
    if [[ " $CRASHED_TASKS " =~ " $TASK_ID " ]]; then
        RESUME_SCRATCH="${SCRATCH_BASE}/airevolve_resume_${RESUME_JOB}_${TASK_ID}/results/${RUN_NAME}_resumed"

        if [ -d "$RESUME_SCRATCH" ]; then
            # Graft rl_logs
            if [ -d "${RESUME_SCRATCH}/rl_logs/generation_01" ]; then
                mkdir -p "${TARGET_PATH}/rl_logs"
                cp -r "${RESUME_SCRATCH}/rl_logs/generation_01" "${TARGET_PATH}/rl_logs/generation_40"
                echo "  -> Grafted rl_logs/generation_40"
            fi

            # Graft snapshots
            if [ -d "${RESUME_SCRATCH}/snapshots/gen_001" ]; then
                mkdir -p "${TARGET_PATH}/snapshots"
                cp -r "${RESUME_SCRATCH}/snapshots/gen_001" "${TARGET_PATH}/snapshots/gen_040"
                
                # Update the generation number in the pareto_info.csv to 40
                CSV_FILE="${TARGET_PATH}/snapshots/gen_040/pareto_info.csv"
                if [ -f "$CSV_FILE" ]; then
                    awk -F, 'BEGIN{OFS=","} NR==1{print $0} NR>1{$2=40; print $0}' "$CSV_FILE" > "${CSV_FILE}.tmp" && mv "${CSV_FILE}.tmp" "$CSV_FILE"
                fi
                echo "  -> Grafted snapshots/gen_040 and updated CSV generation column"
            fi
        else
            echo "  [Warning] Resumed scratch dir not found for crashed task: $RESUME_SCRATCH"
        fi
    fi
done

echo "Unification complete!"