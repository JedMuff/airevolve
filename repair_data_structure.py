import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Adjust the python path to import the sorting utilities from your local airevolve repo
REPO_ROOT = Path("/Users/mikolajduchlinski/Desktop/airevolve")
sys.path.insert(0, str(REPO_ROOT))

from airevolve.evolution_tools.selectors.nsga2_utils import (
    fast_non_dominated_sort,
    calculate_crowding_distance,
)

BASE_DIR = Path("/Users/mikolajduchlinski/Desktop/results_final/results")

RUN_DIRS = [
    "exp_standard_ppo_power_ea_figure8_rep1",
    "exp_standard_ppo_power_ea_figure8_rep2",
    "exp_standard_ppo_power_ea_figure8_rep3",
    "exp_standard_ppo_power_ea_figure8_rep4",
    "exp_standard_ppo_power_ea_figure8_rep5",
    "exp_standard_ppo_power_ea_shuttlerun_rep1",
    "exp_standard_ppo_power_ea_shuttlerun_rep2",
    "exp_standard_ppo_power_ea_shuttlerun_rep3",
    "exp_standard_ppo_power_ea_shuttlerun_rep4",
    "exp_standard_ppo_power_ea_shuttlerun_rep5",
]

def compute_nsga2_ranks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rank"] = -1
    df["crowding_distance"] = 0.0

    for gen, group in df.groupby("generation"):
        fitnesses = group[["gates_passed", "total_energy_j"]].values
        fronts = fast_non_dominated_sort(fitnesses)

        for rank_idx, front_indices in enumerate(fronts):
            original_indices = group.index[front_indices]
            df.loc[original_indices, "rank"] = rank_idx

            front_fitnesses = fitnesses[front_indices]
            cd = calculate_crowding_distance(front_fitnesses)
            df.loc[original_indices, "crowding_distance"] = cd

    return df

def main():
    print("=" * 60)
    print("RECONSTRUCTING FULL GENERATIONAL HISTORY")
    print(f"Results Directory: {BASE_DIR}")
    print("=" * 60)

    for run_name in RUN_DIRS:
        run_dir = BASE_DIR / run_name
        corr_csv = run_dir / "evolution_data_corrected.csv"
        
        if not corr_csv.exists():
            print(f"  [SKIP] Missing corrected data file for {run_name}")
            continue
            
        print(f"\nProcessing {run_name}...")
        
        # Load the corrected data as our ground truth for fitness evaluations
        df_corr = pd.read_csv(corr_csv)
        df_corr['id'] = df_corr['id'].astype(str).str.zfill(4)
        
        # Build lookup dictionaries from the corrected CSV
        fitness_lookup = {}
        parent_lookup = {}
        for _, row in df_corr.iterrows():
            ind_id = row['id']
            fitness_lookup[ind_id] = (row['gates_passed'], row['total_energy_j'])
            parent_lookup[ind_id] = row['parent_ids']
            
        # Build lookup dictionaries from the file system for genome and log_dir
        # This gracefully handles both intact and crashed runs since the files exist on disk!
        genome_lookup = {}
        log_dir_lookup = {}
        for genome_path in run_dir.glob("rl_logs/generation_*/individual_*/genome.npy"):
            ind_dir = genome_path.parent
            ind_id = ind_dir.name.split("_")[-1]
            try:
                genome = np.load(genome_path, allow_pickle=True)
                genome_lookup[ind_id] = np.array2string(genome, separator=',', max_line_width=np.inf)
            except Exception as e:
                print(f"Error loading {genome_path}: {e}")
                genome_lookup[ind_id] = ""
            log_dir_lookup[ind_id] = str(ind_dir)
            
        reconstructed_rows = []
        
        def make_row(ind_id, gen):
            g, e = fitness_lookup.get(ind_id, (0.0, 1e9))
            return {
                "id": ind_id,
                "generation": gen,
                "genome": genome_lookup.get(ind_id, ""),
                "log_dir": log_dir_lookup.get(ind_id, ""),
                "parent_ids": parent_lookup.get(ind_id, "[]"),
                "in_pop": True,
                "gates_passed": g,
                "total_energy_j": e
            }
        
        # 1. Generation 0: just the 24 initial individuals (no survivors yet)
        gen0_ids = df_corr[df_corr['generation'] == 0]['id'].values
        for ind_id in gen0_ids:
            reconstructed_rows.append(make_row(ind_id, 0))
            
        # 2. Generation 1 to 40: survivors from snapshot + ALL new offspring born that gen
        last_gen_found = 0
        for gen in range(1, 41):
            snapshot_csv = run_dir / "snapshots" / f"gen_{gen:03d}" / "pareto_info.csv"
            if not snapshot_csv.exists():
                break
                
            last_gen_found = gen
            
            # Get the 24 survivor IDs from the snapshot
            df_snap = pd.read_csv(snapshot_csv)
            df_snap['id'] = df_snap['id'].astype(str).str.zfill(4)
            survivor_ids = set(df_snap['id'].values)
            
            # Get ALL new offspring born in this generation from the corrected CSV
            offspring_ids = set(df_corr[df_corr['generation'] == gen]['id'].values)
            
            # Full candidate pool = survivors ∪ offspring (avoid duplicates)
            all_ids_this_gen = survivor_ids | offspring_ids
            
            for ind_id in all_ids_this_gen:
                reconstructed_rows.append(make_row(ind_id, gen))

                
        if last_gen_found == 0:
            print(f"    [WARN] No snapshots found for {run_name}! Skipping reconstruction.")
            continue
            
        # 3. Create the repaired full evolution dataframe
        df_repaired = pd.DataFrame(reconstructed_rows)
        
        # 4. Recompute the NSGA-II ranks per generation using the newly evaluated fitnesses!
        df_repaired = compute_nsga2_ranks(df_repaired)
        
        # 5. Format to exact user specification
        df_repaired["fitness"] = df_repaired.apply(lambda row: f"({row['gates_passed']}, {row['total_energy_j']})", axis=1)
        
        desired_columns = [
            "id", "generation", "genome", "log_dir", "parent_ids", 
            "in_pop", "fitness", "rank", "crowding_distance", 
            "gates_passed", "total_energy_j"
        ]
        df_repaired = df_repaired[desired_columns]
        
        out_csv = run_dir / "evolution_data_repaired.csv"
        df_repaired.to_csv(out_csv, index=False)
        print(f"    -> Reconstructed full history (Gen 0 to {last_gen_found}) into evolution_data_repaired.csv ({len(df_repaired)} rows)")
        
        # 6. Extract the final Pareto front from the repaired dataframe
        pareto_final = df_repaired[(df_repaired["generation"] == last_gen_found) & (df_repaired["rank"] == 0)].copy()
        
        # The user explicitly wants pareto_front_final_repaired.csv to be clean and NOT have the genome array!
        clean_columns = [
            "id", "generation", "gates_passed", "total_energy_j", 
            "parent_ids", "rank", "crowding_distance", "fitness"
        ]
        pareto_final_clean = pareto_final[clean_columns]
        
        pareto_csv = run_dir / "pareto_front_final_repaired.csv"
        pareto_final_clean.to_csv(pareto_csv, index=False)
        print(f"    -> Saved {len(pareto_final_clean)} true Pareto individuals to pareto_front_final_repaired.csv")

if __name__ == "__main__":
    main()
