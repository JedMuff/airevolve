import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def create_table(ax, data, title):
    ax.axis('off')
    table = ax.table(cellText=data, colLabels=["Parameter", "Value"], loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Style the table
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#dddddd')
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4c72b0')
        else:
            if row % 2 == 0:
                cell.set_facecolor('#f5f5f5')
            else:
                cell.set_facecolor('#ffffff')
                
    ax.set_title(title, fontweight='bold', fontsize=12, pad=10)

def main():
    hardware_data = [
        ["Target Mass", "295g"],
        ["Motors", "1404 3700KV Brushless"],
        ["Propellers", "3-inch"],
        ["Battery", "Tattu R-Line 750mAh 14.8V 95C 4S"],
        ["Frame Structure", "Carbon fiber tubes (8mm OD, 6mm ID)"],
        ["Max Angular Velocity", "4399 rad/s (42004.88 RPM)"],
        ["Thrust Coef (k_f)", "1.80e-07"],
        ["Torque Coef (k_m)", "2.89e-09"],
    ]
    
    ea_data = [
        ["Experiment repetitions", "5"],
        ["Generations", "32"],
        ["Population Size", "24"],
        ["Number of mutations", "24"],
        ["Genome", "Spherical"],
        ["Morphology Constraint", "Strictly Hexacopters (6 arms), 3 inch prpopellers"],
        ["Initial Pop Mode", "hover_repair, rejecting non-viable drones"],
        ["Mutation Rate", "19% motor & arms angles/arm length, 5% motor spin direction"],
        ["Scale of Mutation", "10% arm length, 20% motor & arms angles"],
        ["EA Evaluation Time", "12 seconds"],
        ["Fitness Objectives", "Maximize gates_passed, Minimize total_energy_j"],
        ["SoC Randomization", "None (Always 100%). For fair candidates selection"],
    ]
    
    rl_data = [
        ["Algorithm", "Standard PPO"],
        ["Gen 0 Training Timesteps", "10 000 000"],
        ["Gen 1+ Lamarckian Timesteps", "2 000 000"],
        ["Fallback Timesteps ", "3 000 000"],
        ["Observation Dimensions", "26-dimensions"],
        ["Kinematics (6-dimensions)", "pos_x, pos_y, pos_z, vel_x, vel_y, vel_z"],
        ["Attitude & Body rates (6-dimensions)", "roll (angle), pitch (angle), yaw (angle) & p (roll_rate), q (pitch rate), r (yaw rate)"],
        ["Motors angular frequency (6-dimensions)", "rpm_1, rpm_2, rpm_3, rpm_4, rpm_5, rpm_6"],
        ["Gates ahead (for each gate) (2 x 4 dimensions)", "pos_x, pos_y, pos_z, yaw (required towards the gate)"],
        ["Gate Config", "figure8"],
        ["Z Drag Multiplier", "5.0"],
        ["Action Smoothness Penalty", "0.003 * ||u_t - u_{t-1}||"]
    ]

    def save_single_table(data, title, filename):
        fig, ax = plt.subplots(figsize=(10, 4))
        create_table(ax, data, title)
        fig.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"Saved {filename}")

    save_single_table(hardware_data, "Hardware & Physics Setup", "experimental_setup_hardware.png")
    save_single_table(ea_data, "Evolutionary Algorithm Setup", "experimental_setup_ea.png")
    save_single_table(rl_data, "Reinforcement Learning Setup", "experimental_setup_rl.png")

if __name__ == "__main__":
    main()
