'''
This files contains an extra tool for debugging the particle repair operator. Useful for visualizing the step-by-step repair process.
'''

from typing import List, Dict, Any, Optional, Tuple, Callable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import copy
from dataclasses import dataclass
from contextlib import contextmanager

# Import AirEvolve modules
from airevolve.evolution_tools.genome_handlers.operators.particle_repair_operator import (
    are_distance_constraints_violated,
    are_there_cylinder_collisions,
    particle_repair_individual
)
from airevolve.evolution_tools.genome_handlers.conversions.arm_conversions import (
    arms_to_cylinders_cartesian_euler,
    cylinders_to_arms_cartesian_euler,
    arms_to_cylinders_polar_angular,
    cylinders_to_arms_polar_angular
)

# Updated imports for new visualization API
from airevolve.evolution_tools.inspection_tools.drone_visualizer import (
    DroneVisualizer, 
    VisualizationConfig, 
    visualize_cylinders,
    compare_cylinder_arrangements
)
from airevolve.evolution_tools.genome_handlers.cartesian_euler_genome_handler import CartesianEulerDroneGenomeHandler
from airevolve.evolution_tools.genome_handlers.spherical_angular_genome_handler import SphericalAngularDroneGenomeHandler
from airevolve.evolution_tools.genome_handlers.operators.repair_base import RepairConfig

from airevolve.evolution_tools.genome_handlers.operators.particle_repair_operator import (
    enforce_distance_constraints,
    resolve_collision,
    get_combinations
)

def debug_repair(genome_handler: CartesianEulerDroneGenomeHandler, 
                 arms_to_cylinders: Callable = arms_to_cylinders_cartesian_euler, 
                 cylinders_to_arms: Callable = cylinders_to_arms_cartesian_euler):
    """
    Debug and visualize the step-by-step repair process using cylinder visualization.
    """
    print("=== Step-by-Step Repair Visualization ===")
    print(f"Symmetry Operator Applied: {genome_handler.symmetry_operator}")
    genome = genome_handler.genome.copy()
    print(f"Initial genome shape: {genome.shape}")
    print(f"Number of arms: {np.sum(~np.isnan(genome).any(axis=1))}")

    # if genome.shape[1] != 7:
    #     raise ValueError("Cartesian genome must have 7 parameters per arm")
    
    result = genome.copy()
    
    # Create visualizer with custom configuration
    config = VisualizationConfig(
        cylinder_color='blue',
        constraint_min_color='red',
        constraint_max_color='green',
        show_constraints=True,
        elevation=20,
        azimuth=45,
        fontsize=10
    )
    visualizer = DroneVisualizer(config)
    
    # Step 1: Initial genome
    initial_cylinders = arms_to_cylinders(genome, 0.0762, 0.3048)
    
    # Unapply symmetry if it was applied
    symmetry_axis = None
    print(f"Symmetry Operator Applied: {genome_handler.symmetry_operator is not None}")
    if genome_handler.symmetry_operator.get_plane() is not None:
        result = genome_handler.symmetry_operator.unapply_symmetry(result)
        if genome_handler.bilateral_plane_for_symmetry == "xy":
            symmetry_axis = [0, 0, 1]
        elif genome_handler.bilateral_plane_for_symmetry == "xz":
            symmetry_axis = [0, 1, 0]
        elif genome_handler.bilateral_plane_for_symmetry == "yz":
            symmetry_axis = [1, 0, 0]
        print(f"Using symmetry axis: {symmetry_axis}")
    
    # Step 2: After unapplying symmetry
    symmetry_cylinders = arms_to_cylinders(result, 0.0762, 0.3048)
    # fig1, axes1 = compare_cylinder_arrangements(
    #     [initial_cylinders, symmetry_cylinders],
    #     labels=["Initial Genome", "After Unapplying Symmetry"],
    #     d_min=0.09,
    #     d_max=0.4,
    #     config=config
    # )
    # fig1.suptitle("Step 1-2: Initial Genome and After Unapplying Symmetry", fontsize=14)

    # Step 4: Clip parameters to bounds
    result = genome_handler.repair_operator._clip_parameters(result)
    genome_handler.genome = result.copy()
    clipped_cylinders = arms_to_cylinders(result, 0.0762, 0.3048)
    
    # fig3, axes3 = compare_cylinder_arrangements(
    #     [symmetry_cylinders, clipped_cylinders],
    #     labels=["Before Clipping", "After Clipping Parameters"],
    #     d_min=0.09,
    #     d_max=0.4,
    #     config=config
    # )
    # fig3.suptitle("Step 4: After Clipping Parameters to Bounds", fontsize=14)
    
    # Step 5: Apply collision repair
    print("Collision Repair Enabled:", genome_handler.repair_operator.config.enable_collision_repair)
    if genome_handler.repair_operator.config.enable_collision_repair:
        result = debug_particle_repair_individual(
            genome_handler.genome,
            propeller_radius=genome_handler.repair_operator.config.propeller_radius,
            inner_boundary_radius=genome_handler.repair_operator.config.inner_boundary_radius,
            outer_boundary_radius=genome_handler.repair_operator.config.outer_boundary_radius,
            max_iterations=genome_handler.repair_operator.config.max_repair_iterations,
            step_size=genome_handler.repair_operator.config.repair_step_size,
            propeller_tolerance=genome_handler.repair_operator.config.propeller_tolerance,
            repair_along_fixed_axis=None,
            arms_to_cylinders=arms_to_cylinders,
            cylinders_to_arms=cylinders_to_arms,
            visualizer=visualizer
        )

    repaired_cylinders = arms_to_cylinders(result, 0.0762, 0.3048)
    # fig4, axes4 = compare_cylinder_arrangements(
    #     [clipped_cylinders, repaired_cylinders],
    #     labels=["Before Collision Repair", "After Collision Repair"],
    #     d_min=0.09,
    #     d_max=0.4,
    #     config=config
    # )
    # fig4.suptitle("Step 5: After Collision Repair", fontsize=14)

    # Step 6: Apply symmetry restoration
    if genome_handler.symmetry_operator.apply_symmetry and genome_handler.symmetry_operator is not None:
        result = genome_handler.repair_operator.symmetry_operator.apply_symmetry(result)
        genome_handler.genome = result.copy()
        result = debug_particle_repair_individual(
            genome_handler.genome,
            propeller_radius=genome_handler.repair_operator.config.propeller_radius,
            inner_boundary_radius=genome_handler.repair_operator.config.inner_boundary_radius,
            outer_boundary_radius=genome_handler.repair_operator.config.outer_boundary_radius,
            max_iterations=genome_handler.repair_operator.config.max_repair_iterations,
            step_size=genome_handler.repair_operator.config.repair_step_size,
            propeller_tolerance=genome_handler.repair_operator.config.propeller_tolerance,
            repair_along_fixed_axis=symmetry_axis,
            arms_to_cylinders=arms_to_cylinders,
            cylinders_to_arms=cylinders_to_arms,
            visualizer=visualizer
        )
    
    final_cylinders = arms_to_cylinders(result, 0.0762, 0.3048)
    # fig5, axes5 = compare_cylinder_arrangements(
    #     [repaired_cylinders, final_cylinders],
    #     labels=["Before Symmetry", "After Restoring Symmetry"],
    #     d_min=0.09,
    #     d_max=0.4,
    #     config=config
    # )
    # fig5.suptitle("Step 6: After Restoring Symmetry and Final Collision Repair", fontsize=14)
    
    # Final comparison
    # fig6, axes6 = compare_cylinder_arrangements(
    #     [initial_cylinders, final_cylinders],
    #     labels=["Initial Genome", "Final Repaired Genome"],
    #     d_min=0.09,
    #     d_max=0.4,
    #     config=config
    # )
    # fig6.suptitle("Final Comparison: Initial vs Repaired Genome", fontsize=14)

    # Show all plots
    # plt.show()

    return result

def debug_particle_repair_individual(individual, propeller_radius, inner_boundary_radius, outer_boundary_radius,
                     max_iterations=25, step_size=1.0, propeller_tolerance=0.1, repair_along_fixed_axis=None,
                     arms_to_cylinders=arms_to_cylinders_polar_angular, cylinders_to_arms=cylinders_to_arms_polar_angular,
                     visualizer=None):
    """
    Debug particle repair with step-by-step visualization using the new API.
    """
    print(f"Individual:\n{individual}")
    
    # Create visualizer if not provided
    if visualizer is None:
        config = VisualizationConfig(
            cylinder_color='orange',
            constraint_min_color='red',
            constraint_max_color='green',
            show_constraints=True,
            elevation=30,
            azimuth=45
        )
        visualizer = DroneVisualizer(config)
    
    # Initialize - filter valid arms and convert to cylinders
    valid_arms = ~np.isnan(individual.copy()).any(axis=-1)
    valid_arm_params = individual.copy()[valid_arms]
    cylinders = arms_to_cylinders(valid_arm_params)
    num_arms = len(cylinders)
    
    # Precompute constants
    required_clearance = (2 + propeller_tolerance) * propeller_radius

    # Initial state visualization
    # fig_initial, ax_initial = visualizer.plot_cylinders_3d(
    #     cylinders,
    #     title="Initial Genome State",
    #     d_min=inner_boundary_radius,
    #     d_max=outer_boundary_radius
    # )
    
    # Also show interactive view
    # try:
    #     process_initial = visualizer.plot_cylinders_trimesh(cylinders)
    #     print("Interactive 3D viewer opened for initial state")
    # except Exception as e:
    #     print(f"Could not open trimesh viewer: {e}")

    iteration_arrangements = [cylinders.copy()]  # Store each iteration (cylinders are immutable)
    iteration_labels = ["Initial"]

    for iteration in range(max_iterations):
        prev_cylinders = cylinders.copy()  # Cylinders are immutable, shallow copy is sufficient
        
        # Apply distance constraints
        constraints_violated = are_distance_constraints_violated(
            cylinders,
            inner_boundary_radius=inner_boundary_radius, 
            outer_boundary_radius=outer_boundary_radius
        )
        if constraints_violated:
            cylinders = enforce_distance_constraints(cylinders, inner_boundary_radius, outer_boundary_radius)
        
        # Check and resolve all collisions
        collision_pairs = get_combinations(num_arms, repair_along_fixed_axis)
        collisions_found = False
        for i, j in collision_pairs:
            if are_there_cylinder_collisions([cylinders[i], cylinders[j]]):
                collisions_found = True
                new_cyl_i, new_cyl_j = resolve_collision(cylinders[i], cylinders[j], required_clearance, 
                                                       step_size, repair_along_fixed_axis)
                cylinders[i] = new_cyl_i
                cylinders[j] = new_cyl_j
        
        # Store this iteration's result (cylinders are already immutable, no need for deepcopy)
        iteration_arrangements.append(cylinders.copy())
        status = []
        if collisions_found:
            status.append("Collisions")
        if constraints_violated:
            status.append("Constraints")
        if not status:
            status.append("Converged")
        iteration_labels.append(f"Iter {iteration + 1}: {', '.join(status)}")
        
        # Visualize the step
        # fig_step, axes_step = compare_cylinder_arrangements(
        #     [prev_cylinders, cylinders],
        #     labels=[f"Before Iter {iteration + 1}", f"After Iter {iteration + 1}"],
        #     d_min=inner_boundary_radius,
        #     d_max=outer_boundary_radius,
        #     config=visualizer.config
        # )
        # fig_step.suptitle(f"Iteration {iteration + 1}: Collisions={collisions_found}, Constraints={constraints_violated}")
        
        # Show interactive view for this iteration
        # try:
        #     process_iter = visualizer.plot_cylinders_trimesh(cylinders)
        #     print(f"Interactive 3D viewer opened for iteration {iteration + 1}")
        # except Exception as e:
        #     print(f"Could not open trimesh viewer for iteration {iteration + 1}: {e}")
        
        # input(f"Press Enter to continue to next iteration ({iteration + 1}/{max_iterations})...")
        # plt.show()
        
        # Check for convergence
        if not (collisions_found or constraints_violated):
            print(f"Converged after {iteration + 1} iterations")
            break
    
    # Show complete iteration sequence
    # if len(iteration_arrangements) > 2:
    #     # Show first few iterations
    #     num_to_show = min(4, len(iteration_arrangements))
    #     fig_sequence, axes_sequence = compare_cylinder_arrangements(
    #         iteration_arrangements[:num_to_show],
    #         labels=iteration_labels[:num_to_show],
    #         d_min=inner_boundary_radius,
    #         d_max=outer_boundary_radius,
    #         figsize=(20, 10),
    #         config=visualizer.config
    #     )
    #     fig_sequence.suptitle("Particle Repair Iteration Sequence", fontsize=16)
        
        # # Show final comparison if we have many iterations
        # if len(iteration_arrangements) > 4:
        #     fig_final_comp, axes_final_comp = compare_cylinder_arrangements(
        #         [iteration_arrangements[0], iteration_arrangements[-1]],
        #         labels=["Initial State", "Final Converged State"],
        #         d_min=inner_boundary_radius,
        #         d_max=outer_boundary_radius,
        #         config=visualizer.config
        #     )
        #     fig_final_comp.suptitle("Final Particle Repair Comparison", fontsize=16)
    
    # Convert back and update individual
    repaired_arms = cylinders_to_arms(cylinders)
    new_individual = individual.copy()
    new_individual[valid_arms, :-1] = repaired_arms
    
    return new_individual
  
if __name__ == "__main__":
    
    for i in range(3):
        genome_handler = SphericalAngularDroneGenomeHandler(
            min_max_narms=(6, 6),  # Fixed 6 arms for testing
            # bilateral_plane_for_symmetry="xz",  # YZ plane symmetry
            repair=False,
            enable_collision_repair=False,
            propeller_radius=0.0762,
            inner_boundary_radius=0.09,
            outer_boundary_radius=0.4,
            append_arm_chance=0.0,  # No appending for this test
        )
        # no_collision_genome = np.array([
        #     [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0],
        #     [0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 1],
        #     [-0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0],
        #     [0.0, -0.3, 0.0, 0.0, 0.0, 0.0, 1],
        # ])
        # Create a sample genome for testing - force some collisions
        genome_handler.genome = genome_handler._generate_random_genome()

        # genome_handler.genome = no_collision_genome
        print(f"Generated genome shape: {genome_handler.genome.shape}")
        print(f"Initial genome:\n{genome_handler.genome}")
        
        new_genome_handler = SphericalAngularDroneGenomeHandler(
            genome=genome_handler.genome.copy(),
            min_max_narms=(6, 6),  # Fixed 6 arms for testing
            # bilateral_plane_for_symmetry="xz",  # YZ plane symmetry
            repair=True,
            enable_collision_repair=True,
            propeller_radius=0.0762,
            inner_boundary_radius=0.09,
            outer_boundary_radius=0.4,
            append_arm_chance=0.0,  # No appending for this test
        )

        new_genome_handler.genome = genome_handler.genome.copy()

        print("\n" + "="*60)
        print("Starting Debug Repair Process")
        print("="*60)
        
        result = debug_repair(new_genome_handler, arms_to_cylinders=arms_to_cylinders_polar_angular, cylinders_to_arms=cylinders_to_arms_polar_angular)
        
        print("\n" + "="*60)
        print("Debug Repair Process Complete")
        print("="*60)
        print(f"Final genome:\n{result}")
        
        # plt.show()  # Show any remaining visualizations