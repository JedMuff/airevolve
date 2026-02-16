"""
Main script to generate all drone parts.
Runs all part generation scripts in sequence for all arms.
"""

import sys
import os
import importlib

# Add the parent directory to the path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daedalus_forge import config

print("=" * 70)
print("DAEDALUS FORGE - Drone Part Generator")
print("=" * 70)
print(f"Generating parts for {len(config.individual_parameter_vector)} arms")
print()

# Generate parts for each arm
for arm_idx in range(len(config.individual_parameter_vector)):
    print("=" * 70)
    print(f"ARM {arm_idx + 1}")
    print("=" * 70)

    # Override sys.argv to pass arm index to the scripts
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0], str(arm_idx)]

    # Generate arm attachment mount
    print("\nGenerating arm attachment mount...")
    print("-" * 70)
    # Reload module to re-execute with new arm_index
    if 'daedalus_forge.arm_attachement_mount' in sys.modules:
        importlib.reload(sys.modules['daedalus_forge.arm_attachement_mount'])
    else:
        from daedalus_forge import arm_attachement_mount
    print()

    # Generate motor arm attachment
    print("Generating motor arm attachment...")
    print("-" * 70)
    # Reload module to re-execute with new arm_index
    if 'daedalus_forge.motor_arm_attachment' in sys.modules:
        importlib.reload(sys.modules['daedalus_forge.motor_arm_attachment'])
    else:
        from daedalus_forge import motor_arm_attachment
    print()

    # Restore original argv
    sys.argv = original_argv

# Generate landing leg (only once, not arm-specific)
print("=" * 70)
print("LANDING LEG")
print("=" * 70)
print("\nGenerating landing leg...")
print("-" * 70)
from daedalus_forge import landing_leg
print()

print("=" * 70)
print("ALL PARTS GENERATED SUCCESSFULLY!")
print("=" * 70)
print(f"\nGenerated parts for {len(config.individual_parameter_vector)} arms plus landing leg")
print("\nCheck the following locations for output files:")
print("  - STEP files: Current directory")
print("  - STL files: daedalus_forge/stl/")
